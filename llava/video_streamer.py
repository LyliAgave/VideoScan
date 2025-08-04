from .mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from .constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from .conversation import conv_templates, SeparatorStyle
from .model import *

import copy
import torch

from copy import deepcopy

from transformers.cache_utils import Cache

from decord import VideoReader, cpu

import numpy as np

def load_video(video_path, max_frames_num, fps, force_sample):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

    avg_fps = vr.get_avg_fps()

    total_frame_num = len(vr)
    total_video_time = total_frame_num / avg_fps
    fps = round(avg_fps/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_times = [i/fps for i in frame_idx]

    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_times = [i/avg_fps for i in frame_idx]
        
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames, frame_times, total_video_time, avg_fps


class VideoStreamer:
    def __init__(self, video_paths, image_processor, torch_device="cuda", 
                 max_frames_num=64, fps=1, force_sample=False, stream_inf=False, add_time_instruction=False, kv_select_layer=None,
                 **kwargs) -> None:
        self.conv_template = "qwen_1_5"
        self.image_processor = image_processor
        
        self.video_paths = video_paths
        self.batch_size = len(video_paths)

        # get batch video data
        self.frame_times = []
        self.total_video_times = []
        if stream_inf:
            self.video = None
        else:
            self.videos = []
            for i in range(len(video_paths)):
                video, frame_times, total_video_time, _ = load_video(video_paths[i], max_frames_num, fps, force_sample)
                video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
                self.videos.append(video)
                self.frame_times.append(frame_times)
                self.total_video_times.append(total_video_time)
            self.frames_num = len(self.frame_times[0])
        
        # model config
        self.device = torch_device
        # self.eos_token_id = self.model.config.eos_token_id
        self.do_sample = False
        if force_sample:
            self.do_sample = True
        self.add_time_instruction = True
        self.kv_select_layer = kv_select_layer
        
        # vision config
        self.modalities = ["video" for i in range(self.batch_size)]


    def get_cur_query(self, frame_idx, b_idx, question):
        if self.add_time_instruction:
            frame_times = self.frame_times[b_idx]
            video_time = frame_times[frame_idx]
            frame_times = ",".join([f"{i:.2f}s" for i in frame_times[:frame_idx + 1]])
            time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {frame_idx+1} frames are uniformly sampled from it. These frames are located at {frame_times}.Please answer the following questions related to this video."
            template_question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n" + f"{question}"
        else:
            template_question = DEFAULT_IMAGE_TOKEN + f"\n{question}"
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], template_question)
        conv.append_message(conv.roles[1], None)
        self.conv = conv

        prompt_question = conv.get_prompt()

        return prompt_question, template_question


    def select_kv_by_layer(self, past_key_values, attn_scores, batch_split, batched_component_size, top_k=16, drop_pos=None, score_layer_idx=0):
        new_past_key_values = []
        score_layer_attn = attn_scores[int(score_layer_idx)] 
        for _, layer_kv in enumerate(past_key_values):
            new_layer_kv = []
            for k_or_v in layer_kv:
                new_k_or_v = []
                for bb, (sys_len, query_len, image_len) in enumerate(batch_split):
                    left_kv = k_or_v[bb, :, :sys_len].unsqueeze(0)
                    if drop_pos is not None and len(drop_pos) > 0:
                        pos = drop_pos[bb]
                        mask = torch.ones(sys_len, dtype=torch.bool)
                        mask[pos + batched_component_size[0][0]] = False
                        left_kv = left_kv[:, :, mask, :]
                    attn = score_layer_attn[bb, :, -query_len-1, -query_len-image_len:-query_len-1]
                    attn = attn.mean(dim=0) 
                    _, topk_indices = torch.topk(attn, top_k, dim=-1)
                    topk_indices = torch.sort(topk_indices, dim=-1)[0]
                    cur_frame = k_or_v[bb, :, -query_len-image_len:-query_len-1]
                    selected_kv = cur_frame[:, topk_indices].unsqueeze(0)
                    right_kv = k_or_v[bb, :, -query_len-1:-query_len].unsqueeze(0)
                    new_k_or_v.append(torch.cat((left_kv, selected_kv, right_kv), dim=-2))
                new_layer_kv.append(torch.cat(new_k_or_v))
            new_past_key_values.append(new_layer_kv)

        return new_past_key_values


    # select kv by attention score
    def set_w_more_kv(self, past_key_values, attn_scores, batch_split, batched_component_size, top_k=16, drop_pos=None):
        if self.kv_select_layer is not None:
            return self.select_kv_by_layer(past_key_values, attn_scores, batch_split, batched_component_size, top_k=top_k, drop_pos=drop_pos, score_layer_idx=self.kv_select_layer)
        new_past_key_values = []
        for layer_idx, layer_kv in enumerate(past_key_values):
            attn_score = attn_scores[layer_idx]
            new_layer_kv = []
            for k_or_v in layer_kv:
                new_k_or_v = []
                for bb, (sys_len, query_len, image_len) in enumerate(batch_split):
                    left_kv = k_or_v[bb, :, :sys_len].unsqueeze(0)
                    if drop_pos is not None and len(drop_pos) > 0:
                        pos = drop_pos[bb]
                        mask = torch.ones(sys_len, dtype=torch.bool)
                        mask[pos + batched_component_size[0][0]] = False
                        left_kv = left_kv[:, :, mask, :]
                    attn = attn_score[bb, :, -query_len-1, -query_len-image_len:-query_len-1]
                    attn = attn.mean(dim=0)
                    _, topk_indices = torch.topk(attn, top_k, dim=-1)
                    topk_indices = torch.sort(topk_indices, dim=-1)[0]
                    cur_frame = k_or_v[bb, :, -query_len-image_len:-query_len-1]
                    selected_kv = cur_frame[:, topk_indices].unsqueeze(0)
                    right_kv = k_or_v[bb, :, -query_len-1:-query_len].unsqueeze(0)
                    new_k_or_v.append(torch.cat((left_kv, selected_kv, right_kv), dim=-2))
                new_layer_kv.append(torch.cat(new_k_or_v))
            new_past_key_values.append(new_layer_kv)
        return new_past_key_values
        

    def set_past_key_values(self, trans_past_key_values, batched_component_size, trans_inputs_embeds=None, drop_pos=None,           
                            attentions=None, add_kv=0):
        past_key_values = deepcopy(trans_past_key_values)
        batch_split = []
        for split_size in batched_component_size:
            if len(split_size) > 3:
                batch_split.append([split_size[0]+(split_size[-1]*(1+add_kv)), split_size[1], split_size[2]])
            else:
                batch_split.append(split_size)
        
        new_past_key_values = []
        if attentions is not None:
            new_past_key_values = self.set_w_more_kv(past_key_values, attentions, batch_split, batched_component_size, top_k=add_kv, drop_pos=drop_pos)
        else:
            for layer_kv in past_key_values:
                new_layer_kv = []
                for k_or_v in layer_kv:
                    new_k_or_v = []
                    for bb in range(len(batch_split)):
                        left_kv = k_or_v[bb, :, :batch_split[bb][0]].unsqueeze(0)
                        if drop_pos is not None and len(drop_pos) > 0:
                            pos = drop_pos[bb]
                            mask = torch.ones(batch_split[bb][0], dtype=torch.bool)
                            mask[pos + batched_component_size[0][0]] = False
                            left_kv = left_kv[:, :, mask, :]
                        right_kv = k_or_v[bb, :, -batch_split[bb][1]-1:-batch_split[bb][1]].unsqueeze(0)
                        new_k_or_v.append(torch.cat((left_kv, right_kv), dim=-2))
                    new_layer_kv.append(torch.cat(new_k_or_v))
                new_past_key_values.append(new_layer_kv)

        if trans_inputs_embeds is not None:
            inputs_embeds = deepcopy(trans_inputs_embeds)
            new_inputs_embeds = []
            for bb in range(len(batch_split)):
                new_inputs_embeds.append(inputs_embeds[bb, -batch_split[bb][1]:, :].unsqueeze(0))
            new_inputs_embeds = torch.cat(new_inputs_embeds, dim=0)
            return new_past_key_values, new_inputs_embeds
        return new_past_key_values, None


    @torch.no_grad()
    def stream_generate_only_tok(
            self, 
            model, 
            tokenizer,
            question,
            use_carrier=True,
            **gen_kwargs
        ):
        """
        only when the last frame will model do generate and output
        for previous frames, only prefill will be executed
        """
        past_key_values = None
        past_frames_embeds = [None for i in range(self.batch_size)]
        for cur_frame_idx in range(self.frames_num):
            cur_input_ids = []
            stopping_criterias = []
            for b_idx in range(self.batch_size):
                cur_query, cur_prompt = self.get_cur_query(cur_frame_idx, b_idx, question)
                cur_input_id = tokenizer_image_token(cur_query, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
                stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
                keywords = [stop_str]
                cur_stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, cur_input_id)
                stopping_criterias.append(cur_stopping_criteria)
                cur_input_ids.append(cur_input_id)
            
            max_batch_len = max(x.shape[1] for x in cur_input_ids)
            cur_input_ids_padded = []
            cur_attention_mask = torch.zeros((self.batch_size, max_batch_len), dtype=torch.bool, device=self.device)
            for i, (cur_ipt_ids, cur_attn_mask) in enumerate(zip(cur_input_ids, cur_attention_mask)):
                cur_len = cur_ipt_ids.shape[1]
                cur_input_ids_padded.append(torch.cat((cur_ipt_ids, torch.zeros((1, max_batch_len-cur_len), dtype=cur_ipt_ids.dtype, device=cur_ipt_ids.device)), dim=1))
                cur_attention_mask[i, :cur_len] = True
                
            cur_input_ids_padded = torch.cat(cur_input_ids_padded)
            
            del cur_ipt_ids

            cur_videos = [video[cur_frame_idx] for video in self.videos]
            
            (cur_input_ids, cur_position_ids, cur_attention_mask, past_frames_embeds, cur_batched_split_size, cur_inputs_embeds, _, _) = model.prepare_inputs_wo_last_frame(cur_input_ids_padded, cur_videos, attention_mask=cur_attention_mask, past_frames_embeds=past_frames_embeds, modalities=self.modalities,
                                                                                                                                                                            use_carrier=use_carrier, cur_frame_idx=cur_frame_idx)
            torch.cuda.empty_cache()
            del cur_input_ids_padded
            """
            copy and modify from modeling_qwen.prepare_inputs_for_generation
            """
            if cur_attention_mask is not None and cur_position_ids is None:
                cur_position_ids = cur_attention_mask.long().cumsum(-1) - 1
                cur_position_ids.masked_fill_(cur_attention_mask == 0, 1)
            if past_key_values is not None:
                if isinstance(past_key_values, Cache):
                    cache_length = past_key_values.get_seq_length()
                    past_length = past_key_values.seen_tokens
                    max_cache_length = past_key_values.get_max_length()
                else:
                    cache_length = past_length = past_key_values[0][0].shape[2] 
                    max_cache_length = None
                if cur_attention_mask is not None:
                    cur_position_ids = cur_position_ids[:, -(cur_attention_mask.shape[1] - cache_length) :]
                if cur_inputs_embeds is not None and cur_attention_mask is not None:
                    cur_inputs_embeds = cur_inputs_embeds[:, -(cur_attention_mask.shape[1] - cache_length) :, :]
            
            cur_outputs = model(
                position_ids=cur_position_ids, 
                attention_mask=cur_attention_mask,
                inputs_embeds=cur_inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
            )

            del cur_position_ids
            del cur_attention_mask

            if cur_frame_idx != self.frames_num - 1:
                past_key_values, _ = self.set_past_key_values(cur_outputs.past_key_values, cur_batched_split_size)
            else:
                past_key_values, tmp_inputs_embeds = self.set_past_key_values(cur_outputs.past_key_values, cur_batched_split_size, trans_inputs_embeds=cur_inputs_embeds)
            
            del cur_outputs
            del cur_inputs_embeds
            
            if cur_frame_idx == self.frames_num - 1:
                del past_frames_embeds
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = None
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1
                
                model.config.return_dict_in_generate = True

                final_seq_len = cur_batched_split_size[0][0] + cur_batched_split_size[0][1] + cur_batched_split_size[0][-1] + 1
                cur_attention_mask = torch.ones((self.batch_size, final_seq_len), dtype=torch.bool, device=self.device)

                cur_outputs = model.stream_generate(
                    attention_mask=cur_attention_mask,
                    inputs_embeds=tmp_inputs_embeds,
                    past_key_values=past_key_values,
                    stopping_criteria=stopping_criterias,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )
                del tmp_inputs_embeds
                del past_key_values
                outputs = tokenizer.batch_decode(cur_outputs[0], skip_special_tokens=True)[0].strip()
                del cur_outputs
                return outputs, cur_prompt
            

    @torch.no_grad()
    def vscan_add_kv(
            self, 
            model, 
            tokenizer,
            question,
            use_carrier=True,
            add_kv=16,
            **gen_kwargs
        ):
        past_key_values = None
        past_frames_embeds = [None for i in range(self.batch_size)]
        for cur_frame_idx in range(self.frames_num):
            cur_input_ids = []
            stopping_criterias = []
            for b_idx in range(self.batch_size):
                cur_query, cur_prompt = self.get_cur_query(cur_frame_idx, b_idx, question)
                cur_input_id = tokenizer_image_token(cur_query, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
                stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
                keywords = [stop_str]
                cur_stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, cur_input_id)
                stopping_criterias.append(cur_stopping_criteria)
                cur_input_ids.append(cur_input_id)
            
            max_batch_len = max(x.shape[1] for x in cur_input_ids)
            cur_input_ids_padded = []
            cur_attention_mask = torch.zeros((self.batch_size, max_batch_len), dtype=torch.bool, device=self.device)
            for i, (cur_ipt_ids, cur_attn_mask) in enumerate(zip(cur_input_ids, cur_attention_mask)):
                cur_len = cur_ipt_ids.shape[1]
                cur_input_ids_padded.append(torch.cat((cur_ipt_ids, torch.zeros((1, max_batch_len-cur_len), dtype=cur_ipt_ids.dtype, device=cur_ipt_ids.device)), dim=1))
                cur_attention_mask[i, :cur_len] = True
                
            cur_input_ids_padded = torch.cat(cur_input_ids_padded)
            
            del cur_ipt_ids

            cur_videos = [video[cur_frame_idx] for video in self.videos]
            
            (cur_input_ids, cur_position_ids, cur_attention_mask, past_frames_embeds, cur_batched_split_size, cur_inputs_embeds, _, _) = model.prepare_inputs_wo_last_frame(cur_input_ids_padded, cur_videos, attention_mask=cur_attention_mask, past_frames_embeds=past_frames_embeds, modalities=self.modalities,
                                                                                                                                                                            use_carrier=use_carrier, cur_frame_idx=cur_frame_idx, add_kv=add_kv)
            
            torch.cuda.empty_cache()
            del cur_input_ids_padded
            """
            copy and modify from modeling_qwen.prepare_inputs_for_generation
            """
            if cur_attention_mask is not None and cur_position_ids is None:
                cur_position_ids = cur_attention_mask.long().cumsum(-1) - 1
                cur_position_ids.masked_fill_(cur_attention_mask == 0, 1)
            if past_key_values is not None:
                if isinstance(past_key_values, Cache):
                    cache_length = past_key_values.get_seq_length()
                    past_length = past_key_values.seen_tokens
                    max_cache_length = past_key_values.get_max_length()
                else:
                    cache_length = past_length = past_key_values[0][0].shape[2] 
                    max_cache_length = None
                if cur_attention_mask is not None:
                    cur_position_ids = cur_position_ids[:, -(cur_attention_mask.shape[1] - cache_length) :]
                if cur_inputs_embeds is not None and cur_attention_mask is not None:
                    cur_inputs_embeds = cur_inputs_embeds[:, -(cur_attention_mask.shape[1] - cache_length) :, :]
            
            cur_outputs = model(
                position_ids=cur_position_ids, 
                attention_mask=cur_attention_mask,
                inputs_embeds=cur_inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
                output_attentions=True,
            )

            del cur_position_ids
            del cur_attention_mask

            if cur_frame_idx != self.frames_num - 1:
                past_key_values, _ = self.set_past_key_values(cur_outputs.past_key_values, cur_batched_split_size, attentions=cur_outputs.attentions, add_kv=add_kv)
            else:
                past_key_values, tmp_inputs_embeds = self.set_past_key_values(cur_outputs.past_key_values, cur_batched_split_size, trans_inputs_embeds=cur_inputs_embeds, attentions=cur_outputs.attentions, add_kv=add_kv)
            
            del cur_outputs
            del cur_inputs_embeds
            
            if cur_frame_idx == self.frames_num - 1:
                del past_frames_embeds
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = None
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1
                
                model.config.return_dict_in_generate = True

                final_seq_len = cur_batched_split_size[0][0] + cur_batched_split_size[0][1] + (cur_batched_split_size[0][-1] + 1) * (1 + add_kv)
                cur_attention_mask = torch.ones((self.batch_size, final_seq_len), dtype=torch.bool, device=self.device)

                cur_outputs = model.stream_generate(
                    attention_mask=cur_attention_mask,
                    inputs_embeds=tmp_inputs_embeds,
                    past_key_values=past_key_values,
                    stopping_criteria=stopping_criterias,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )

                del tmp_inputs_embeds
                del past_key_values
                outputs = tokenizer.batch_decode(cur_outputs[0], skip_special_tokens=True)[0].strip()
                del cur_outputs
                return outputs, cur_prompt
                
                
    @torch.no_grad()
    def stream_generate_inf(
            self, 
            model, 
            tokenizer,
            question,
            use_carrier=True,
            **gen_kwargs
        ):
        """
        only when the last frame will model do generate and output
        for previous frames, only prefill will be executed
        """
        model.config.stream_post_model = False

        past_key_values = None
        past_frames_embeds = [None for i in range(self.batch_size)]
        picked_frame_idx = []

        vr = VideoReader(self.video_paths[0], ctx=cpu(0), num_threads=1)
        avg_fps = vr.get_avg_fps()
        total_video_time = len(vr) / avg_fps
        if total_video_time > 1280:
            avg_fps *= 5
        elif total_video_time > 900:
            avg_fps *= 4
        elif total_video_time > 600:
            avg_fps *= 3
        elif total_video_time > 300:
            avg_fps *= 2
        avg_fps = round(avg_fps)
        frame_idx = [i for i in range(0, len(vr), avg_fps)]
        frame_times = [i/avg_fps for i in frame_idx]
        total_frame_num = len(frame_idx)
        forget_idx = [False] * total_frame_num
        for cur_frame_idx in range(total_frame_num):
            cur_input_ids = []
            stopping_criterias = []
            for b_idx in range(self.batch_size):
                if cur_frame_idx == total_frame_num-1:
                    picked_frame_times = [t for f, t in zip(forget_idx, frame_times) if f]
                    picked_frame_times.append(frame_times[cur_frame_idx])
                    picked_num = len(picked_frame_times)
                    frame_times = ",".join([f"{i:.2f}s" for i in picked_frame_times])
                    time_instruciton = f"The video lasts for {total_video_time:.2f} seconds, and {picked_num} frames are uniformly sampled from it. These frames are located at {frame_times}.Please answer the following questions related to this video."
                    template_question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n" + f"{question}"
                else:
                    template_question = DEFAULT_IMAGE_TOKEN + f"GO"

                conv = copy.deepcopy(conv_templates[self.conv_template])
                conv.append_message(conv.roles[0], template_question)
                conv.append_message(conv.roles[1], None)

                cur_query = conv.get_prompt()
                cur_prompt = template_question

                cur_input_id = tokenizer_image_token(cur_query, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                cur_stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, cur_input_id)
                stopping_criterias.append(cur_stopping_criteria)
                cur_input_ids.append(cur_input_id)
            
            max_batch_len = max(x.shape[1] for x in cur_input_ids)
            cur_input_ids_padded = []
            cur_attention_mask = torch.zeros((self.batch_size, max_batch_len), dtype=torch.bool, device=self.device)
            for i, (cur_ipt_ids, cur_attn_mask) in enumerate(zip(cur_input_ids, cur_attention_mask)):
                cur_len = cur_ipt_ids.shape[1]
                cur_input_ids_padded.append(torch.cat((cur_ipt_ids, torch.zeros((1, max_batch_len-cur_len), dtype=cur_ipt_ids.dtype, device=cur_ipt_ids.device)), dim=1))
                cur_attention_mask[i, :cur_len] = True
                
            cur_input_ids_padded = torch.cat(cur_input_ids_padded)
            del cur_input_ids

            cur_videos = vr.get_batch([frame_idx[cur_frame_idx]]).asnumpy()
            cur_videos = self.image_processor.preprocess(cur_videos, return_tensors="pt")["pixel_values"].cuda().bfloat16()
            cur_videos = [cur_videos]
            forget_idx[cur_frame_idx] = True

            (cur_input_ids, cur_position_ids, cur_attention_mask, past_frames_embeds, cur_batched_split_size, cur_inputs_embeds, _, drop_pos) = model.prepare_inputs_wo_last_frame(cur_input_ids_padded, cur_videos, attention_mask=cur_attention_mask, past_frames_embeds=past_frames_embeds, modalities=self.modalities,
                                                                                                                                                                                    use_carrier=use_carrier, cur_frame_idx=cur_frame_idx, forget_idx=forget_idx)
            
            del cur_input_ids_padded
            torch.cuda.empty_cache()

            """
            copy and modify from modeling_qwen.prepare_inputs_for_generation
            """
            if cur_attention_mask is not None and cur_position_ids is None:
                cur_position_ids = cur_attention_mask.long().cumsum(-1) - 1
                cur_position_ids.masked_fill_(cur_attention_mask == 0, 1)
            if past_key_values is not None:
                if isinstance(past_key_values, Cache):
                    cache_length = past_key_values.get_seq_length()
                    past_length = past_key_values.seen_tokens
                    max_cache_length = past_key_values.get_max_length()
                else:
                    cache_length = past_length = past_key_values[0][0].shape[2] 
                    max_cache_length = None
                if cur_attention_mask is not None:
                    cur_position_ids = cur_position_ids[:, -(cur_attention_mask.shape[1] - cache_length) :]
                if cur_inputs_embeds is not None and cur_attention_mask is not None:
                    cur_inputs_embeds = cur_inputs_embeds[:, -(cur_attention_mask.shape[1] - cache_length) :, :]
            
            cur_outputs = model(
                position_ids=cur_position_ids, 
                attention_mask=cur_attention_mask,
                inputs_embeds=cur_inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
            )

            if cur_frame_idx != total_frame_num - 1:
                past_key_values, tmp_inputs_embeds = self.set_past_key_values(cur_outputs.past_key_values, cur_batched_split_size, drop_pos=drop_pos)
            else:
                past_key_values, tmp_inputs_embeds = self.set_past_key_values(cur_outputs.past_key_values, cur_batched_split_size, trans_inputs_embeds=cur_inputs_embeds)
            
            del cur_outputs
            del cur_inputs_embeds
            del cur_attention_mask
            del cur_position_ids
            
            if cur_frame_idx == total_frame_num - 1:
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = None
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1
                
                model.config.return_dict_in_generate = True

                cur_outputs = model.stream_generate(
                    inputs_embeds=tmp_inputs_embeds,
                    past_key_values=past_key_values,
                    stopping_criteria=stopping_criterias,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )

                del tmp_inputs_embeds
                del past_frames_embeds
                del past_key_values

                outputs = tokenizer.batch_decode(cur_outputs[0], skip_special_tokens=True)[0].strip()

                del cur_outputs

                return outputs, cur_prompt