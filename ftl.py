from transformers import AutoModelForCausalLM, AutoTokenizer
from models.resunet_stft import SealGPT_Sep
import torch
import torchaudio
import torch.nn as nn
import re

class LALM_Filter(nn.Module):
    
    def __init__(self, separator_skpt, llm_name, device='cuda:0', target_sr=16000):
        super().__init__()
        # llm_name: "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"

        self.device = device
        self.target_sr = target_sr

        self.separator = SealGPT_Sep(output_channels=1, dprnn=False)
        self.sep_state_dict = torch.load(separator_skpt, map_location="cpu")
        self.sep_state_dict = self.sep_state_dict["generator"]
        self.separator.load_state_dict(self.sep_state_dict)
        self.separator = self.separator.to(device)
        self.separator.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            dtype="auto",
            device_map= "auto" # {"": self.device}
        )

    @torch.no_grad()
    def forward(self, user_instruction, audio_path):
        prompt = (
        "You are an expert in audio understanding and multimodal reasoning."
        " Your task is to decide what audio input should be provided to a Large Audio Language Model (LALM) in order to best accomplish a user’s instruction."
        " The audio has been separated into two tracks:"
        " speech: contains spoken voice content only;"
        " non-speech: contains non-speech acoustic events only."
        " Mixture refers to the original unseparated audio."
        " You should select the input that maximizes task-relevant information, based on the user’s instruction."
        " Guidelines:"
        " 1. You should ONLY choose ‘speech’ when speech information alone is clearly sufficient to solve the task, AND non-speech provides no meaningful additional information."
        " 2. You should ONLY choose ‘non-speech’ when non-speech audio alone is clearly sufficient to solve the task, AND speech provides no meaningful additional information."
        " 3. In ALL other cases, including uncertainty, partial usefulness of both modalities, or when you cannot strictly rule out one modality, you MUST choose ‘mixture’."
        " Additional Domain Rules:"
        " - Speech is required for linguistic content, speaker intent, emotion, or dialogue understanding."
        " - Non-speech includes environmental sounds and vocal non-linguistic sounds (e.g., laughter, sneeze, cough)."
        " Respond with only one word: speech, non-speech, or mixture. Do not provide explanations."
        f" User Instruction: [{user_instruction}]"
        )

        # print(prompt)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)

        # conduct text completion
        generated_ids = self.llm_model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        content = content.lower()
        content = re.sub(r"[^a-z\s-]", "", content) # keep a-z and '-'

        # separation
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] >= 2:
            audio = torch.mean(audio, dim=0, keepdim=True)
        # print(audio.shape, sr)
        audio = audio.to(self.device)
        # audio = audio[:, :160000]

        if sr != self.target_sr:
            audio = torchaudio.functional.resample(
                audio, orig_freq=sr, new_freq=self.target_sr
            )

        if audio.shape[-1] > int(self.target_sr * 10):
            sep_outputs = self.separator.chunk_inference({"mixture": audio[:,None,:].float()}, NL=1.0, NC=8.0, NR=1.0, sr=self.target_sr)
            sep_speech = torch.from_numpy(sep_outputs[0, :][None, :]).to(self.device)
            sep_sound = torch.from_numpy(sep_outputs[1, :][None, :]).to(self.device)
        else:
            sep_outputs = self.separator({"mixture": audio[:,None,:].float()})
            sep_speech = sep_outputs["pred_speech"][:,0,:]
            sep_sound = sep_outputs["pred_sound"][:,0,:]
        
        
        return sep_speech, sep_sound, audio, content

if __name__ == "__main__":
    separator_skpt = './demo/best_separator_on_valid.bin'
    llm_name = 'Qwen/Qwen3-8B'
    text_prompt = "What is the person talking about?" 
    # "Which kind of animal sound can you hear from the audio?" # "What is the person talking about?" # ...
    audio_path = "./demo/KilQtE5Nl90_0_10_0dB_mixed.wav"
    # "./demo/KilQtE5Nl90_0_10_0dB_mixed.wav"
    # "./demo/J0Ruo0PDfQo_8_18_5dB_mixed.wav"
    out_audio_folder = './demo'

    lalm_filter = LALM_Filter(separator_skpt = separator_skpt, llm_name=llm_name)
    sep_speech, sep_sound, audio, content = lalm_filter(user_instruction=text_prompt, audio_path=audio_path)

    print(f' === User Instruction is: {text_prompt}; Agent ({llm_name}) output is: {content}. ===')
    print(audio.shape, sep_speech.shape, sep_sound.shape)

    torchaudio.save(
            f"{out_audio_folder}/{audio_path.split('/')[-1].replace('.wav','-sep-speech.wav')}",
            sep_speech.detach().cpu(),
            sample_rate=lalm_filter.target_sr
    )

    torchaudio.save(
            f"{out_audio_folder}/{audio_path.split('/')[-1].replace('.wav','-sep-sound.wav')}",
            sep_sound.detach().cpu(),
            sample_rate=lalm_filter.target_sr
    )
    

    