from torch import nn
import torch, torchaudio, torchvision
import torchaudio.functional as audio_F
import torchaudio.transforms as audio_T
import librosa
import numpy as np

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()

    def forward(self, freqs_cis, input_positions, embedder, input_token_ids, hidden_size):
        kv_write_indices = input_positions
        hidden_states = embedder(input_token_ids)
        hidden_states = hidden_states * (hidden_size**0.5)
        return freqs_cis, kv_write_indices, hidden_states
    
class TextDecoder(nn.Module):
    def __init__(self):
        super(TextDecoder, self).__init__()
    def forward(self, embedder, quant, sampler, hidden_states, output_positions, temperatures, top_ps, top_ks):
        embedder_weight = embedder.weight
        if quant:
            embedder_weight = (
                embedder_weight * embedder.weight_scaler.unsqueeze(-1))
        next_tokens = sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        return next_tokens
    

class TextPreprocess(nn.Module):
    def __init__(self):
        super(TextPreprocess, self).__init__()

    def forward(self, prompts, tokenizer, output_len, max_position_embeddings):
        is_str_prompt = isinstance(prompts, str)        
        # If a single prompt is provided, treat it as a batch of 1.
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= max_position_embeddings
        return is_str_prompt, batch_size, prompt_tokens, min_prompt_len, max_seq_len


class TextGenerator(nn.Module):
    def __init__(self):
        super(TextGenerator, self).__init__()
    def forward(self, forward, batch_size, max_seq_len, tokenizer, min_prompt_len, prompt_tokens, device, temperature, top_p, top_k, kv_caches, output_len, is_str_prompt):
        token_ids_tensor = torch.full((batch_size, max_seq_len),
                                            tokenizer.pad_id, dtype=torch.int64)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                            tokenizer.pad_id,
                                            dtype=torch.int64)
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
                p[:min_prompt_len])
        token_ids_tensor = token_ids_tensor.to(device)
        input_token_ids_tensor = input_token_ids_tensor.to(device)
        prompt_mask_tensor = token_ids_tensor != tokenizer.pad_id # 들어있는것만 마스킹
        input_positions_tensor = torch.arange(0, min_prompt_len,
                                            dtype=torch.int64).to(device)
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                                -2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(
            device)
        temperatures_tensor = torch.FloatTensor([temperature] * batch_size).to(
            device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(
            device)

        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.
        for i in range(max_seq_len - min_prompt_len):
            next_token_ids = forward(
                input_token_ids=input_token_ids_tensor,
                input_positions=input_positions_tensor, ## 여기서 포지션 정보 넘어감
                kv_write_indices=None,
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
            )

            curr_prompt_mask = prompt_mask_tensor.index_select(
                1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(
                1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                        next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2,
                                                        input_positions_tensor)
            output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(
                device)
            output_index = output_index + 1
        # Detokenization.
        token_ids = token_ids_tensor.tolist()
        text_results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i])
                                    + output_len]
            ## 종료 조건 -> 잘라내기
            if tokenizer.eos_id in trimmed_output:
                eos_index = trimmed_output.index(tokenizer.eos_id)
                trimmed_output = trimmed_output[:eos_index]
            text_results.append(tokenizer.decode(trimmed_output))
        # If a string was provided as input, return a string as output.
        text_results = text_results[0] if is_str_prompt else text_results
        return text_results


class Encoder1d(nn.Module):
    def __init__(self, resize=None, patch_size=128, embed_dim=2048, model_path=None):
        super(Encoder1d, self).__init__()
        self.resize = resize
        self.patchfier1d = nn.Conv1d(2, embed_dim, patch_size, patch_size)
        self.conv1d = nn.Conv1d(embed_dim, embed_dim, 3, 1, 1)
        if model_path is not None:
            self.load_state_dict(torch.load(model_path), False)

    def forward(self, x, device='cuda', dtype=torch.float):
        x = x.type(dtype).to(device)
        if len(x.shape) != 3:
            x = x.unsqueeze(0)
        if x.shape[-3] < 2:
            x = torch.concatenate([x, x], -2)
        if self.resize is not None:
            x = audio_F.resample(x, x.shape[-1], self.resize, lowpass_filter_width=6)
        x = self.patchfier1d(x)
        residual = x
        x = self.conv1d(x)
        x = x + residual

        return x.permute(0,2,1)
    
class Decoder1d(nn.Module):
    def __init__(self, resize=None, patch_size=128, embed_dim=2048, model_path=None):
        super(Decoder1d, self).__init__()
        self.resize = resize
        self.conv1d = nn.ConvTranspose1d(embed_dim, embed_dim, 3, 1, 1)
        self.patchfier1d = nn.ConvTranspose1d(embed_dim, 2, patch_size, patch_size)
        if model_path is not None:
            self.load_state_dict(torch.load(model_path), False)
        
    def forward(self, x):
        x = x.permute(0,2,1)
        if len(x.shape) != 3:
            x = x.unsqueeze(0)
        residual = x
        x = self.conv1d(x)
        x = x + residual
        x = self.patchfier1d(x)
        if self.resize is not None:
            x = audio_F.resample(x, x.shape[-1], self.resize, lowpass_filter_width=6)
        return x

##수정 필요
class Generator1d(nn.Module):
    def __init__(self):
        super(Generator1d, self).__init__()
    def forward(self, forward, batch_size, data, device, temperature, top_p, top_k, kv_caches, output_len, is_str_prompt):
        print(data.shape)
        mask_tensor = torch.full((1, 1, data.shape[-2], data.shape[-2]),
                                -2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
        # curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        # output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(
        #     device)
        # temperatures_tensor = torch.FloatTensor([temperature] * batch_size).to(
        #     device)
        # top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        # top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        # output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(
        #     device)
        
        # # Prefill up to min_prompt_len tokens, then treat other prefill as
        # # decode and ignore output.
        # for i in range(max_seq_len - min_prompt_len):
        #     next_token_ids = forward(
        #         input_token_ids=input_token_ids_tensor,
        #         input_positions=input_positions_tensor, ## 여기서 포지션 정보 넘어감
        #         kv_write_indices=None,
        #         kv_caches=kv_caches,
        #         mask=curr_mask_tensor,
        #         output_positions=output_positions_tensor,
        #         temperatures=temperatures_tensor,
        #         top_ps=top_ps_tensor,
        #         top_ks=top_ks_tensor,
        #     )

        #     curr_prompt_mask = prompt_mask_tensor.index_select(
        #         1, output_index).squeeze(dim=1)
        #     curr_token_ids = token_ids_tensor.index_select(
        #         1, output_index).squeeze(dim=1)
        #     output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
        #                                 next_token_ids).unsqueeze(dim=1)
        #     token_ids_tensor.index_copy_(1, output_index, output_token_ids)

        #     input_token_ids_tensor = output_token_ids
        #     input_positions_tensor = output_index.unsqueeze(dim=-1)
        #     curr_mask_tensor = mask_tensor.index_select(2,
        #                                                 input_positions_tensor)
        #     output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(
        #         device)
        #     output_index = output_index + 1
        # # Detokenization.
        # token_ids = token_ids_tensor.tolist()
        # text_results = []
        # for i, tokens in enumerate(token_ids):
        #     trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i])
        #                             + output_len]
        #     ## 종료 조건 -> 잘라내기
        #     if tokenizer.eos_id in trimmed_output:
        #         eos_index = trimmed_output.index(tokenizer.eos_id)
        #         trimmed_output = trimmed_output[:eos_index]
        #     text_results.append(tokenizer.decode(trimmed_output))
        # # If a string was provided as input, return a string as output.
        # text_results = text_results[0] if is_str_prompt else text_results
        # return text_results
    

class Encoder2d(nn.Module):
    def __init__(self, resize=None, patch_size=(16,16), embed_dim=2048, model_path=None):
        super(Encoder2d, self).__init__()
        self.resize = resize
        self.embed_dim = embed_dim
        self.feature_shape = None
        self.patchfier2d = nn.Conv2d(4, embed_dim, patch_size, patch_size)
        self.conv2d = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        if model_path is not None:
            self.load_state_dict(torch.load(model_path), False)

    def forward(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(0)
        if x.shape[-3] < 4:
            mean_x = torch.mean(x, -3).unsqueeze(-3).repeat(1,4-x.shape[-3],1,1)
            x = torch.concatenate([x, mean_x], -3)
        if self.resize is not None:
            x = torchvision.transforms.Resize(self.resize)(x)
        x = self.patchfier2d(x)
        residual = x
        x = self.conv2d(x)
        x = x + residual
        self.feature_shape = x.shape

        return x
        # x = x.reshape(1, self.embed_dim, -1)
        # return x.permute(0,2,1)
    
class Decoder2d(nn.Module):
    def __init__(self, resize=None, patch_size=(16,16), embed_dim=2048, model_path=None):
        super(Decoder2d, self).__init__()
        self.resize = resize
        self.embed_dim = embed_dim
        self.feature_shape = None
        self.conv2d = nn.ConvTranspose2d(embed_dim, embed_dim, 3, 1, 1)
        self.patchfier2d = nn.ConvTranspose2d(embed_dim, 4, patch_size, patch_size)
        if model_path is not None:
            self.load_state_dict(torch.load(model_path), False)
        
    def forward(self, x):
        x = x.permute(0,2,1)
        if self.feature_shape is not None:
            x = x.reshape(self.feature_shape)
        residual = x
        x = self.conv2d(x)
        x = x + residual
        x = self.patchfier2d(x)
        if self.resize is not None:
            x = torchvision.transforms.Resize(self.resize)(x)
        return x
    

class Encoder3d(nn.Module):
    def __init__(self, resize=None, patch_size=(6,6,6), embed_dim=2048, model_path=None):
        super(Encoder3d, self).__init__()
        self.resize = resize
        self.embed_dim = embed_dim
        self.feature_shape = None
        self.patchfier3d = nn.Conv3d(4, embed_dim, patch_size, patch_size)
        self.conv3d = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        if model_path is not None:
            self.load_state_dict(torch.load(model_path), False)

    def forward(self, x):
        if len(x.shape) != 5:
            x = x.unsqueeze(0)
        if x.shape[-4] < 4:
            mean_x = torch.mean(x, -4).unsqueeze(-4).repeat(1,4-x.shape[-4],1,1,1)
            x = torch.concatenate([x, mean_x], -4)
        # if self.resize is not None:
        #     x = torchvision.transforms.Resize(self.resize)(x)
        x = self.patchfier3d(x)
        residual = x
        x = self.conv3d(x)
        x = x + residual
        self.feature_shape = x.shape
        x = x.reshape(1, self.embed_dim, -1)
        return x.permute(0,2,1)
    
class Decoder3d(nn.Module):
    def __init__(self, resize=None, patch_size=(6,6,6), embed_dim=2048, model_path=None):
        super(Decoder3d, self).__init__()
        self.resize = resize
        self.embed_dim = embed_dim
        self.feature_shape = None
        self.conv3d = nn.ConvTranspose3d(embed_dim, embed_dim, 3, 1, 1)
        self.patchfier3d = nn.ConvTranspose3d(embed_dim, 4, patch_size, patch_size)
        if model_path is not None:
            self.load_state_dict(torch.load(model_path), False)
        
    def forward(self, x):
        x = x.permute(0,2,1)
        if self.feature_shape is not None:
            x = x.reshape(self.feature_shape)
        residual = x
        x = self.conv3d(x)
        x = x + residual
        x = self.patchfier3d(x)
        # if self.resize is not None:
        #     x = torchvision.transforms.Resize(self.resize)(x)
        return x
    
if __name__ == '__main__':
    print("Start testing...")
    
    ## audio
    test_audio = torch.rand(1, 1, 5200)
    encoder = Encoder1d(resize=520)
    # decoder = Decoder1d(resize=5200)
    # print(decoder(encoder(test_audio)).shape)

    ## image
    # test_image = torch.rand(1, 3, 800, 600)
    # encoder = Encoder2d(resize=(400,300))
    # decoder = Decoder2d(resize=(400,300))
    # feature = encoder(test_image)
    # decoder.feature_shape = encoder.feature_shape
    # print(decoder(feature).shape)

    # video
    test_video = torch.rand(1, 3, 800, 600, 6)
    encoder = Encoder3d(resize=(400,300))
    decoder = Decoder3d(resize=(400,300))
    feature = encoder(test_video)
    decoder.feature_shape = encoder.feature_shape
    print(decoder(feature).shape)
