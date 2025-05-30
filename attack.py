import argparse
import os,json,tqdm,copy
import random,cv2,csv
from transformers import StoppingCriteria, StoppingCriteriaList
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
from video_llama.processors.video_processor import load_video
decord.bridge.set_bridge('torch')

#%%
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='vicuna', help="The type of LLM")
    parser.add_argument("--type", type=str)
    parser.add_argument("--gt_file", type=str)
    parser.add_argument("--video_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--iter", type=int)
    parser.add_argument("--max_modify", type=int)
    parser.add_argument("--step", type=int)
    parser.add_argument("--weight_clip", type=float, default=1)
    parser.add_argument("--weight_llm", type=float, default=1)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args
# def get_boundary():
#     boundary=torch.tensor([[0.0,1.0],[0.0,1.0],[0.0,1.0]])
#     boundary[0]=(boundary[0]-0.48145466)/0.26862954
#     boundary[1]=(boundary[1]-0.4578275)/0.26130258
#     boundary[2]=(boundary[2]-0.40821073)/0.27577711
#     return boundary
def myclamp(x,clean,modif_max):
    #[1,3,8,224,224]
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1, 1).to('cuda')
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1, 1).to('cuda')
    temp = torch.clamp((x+clean) * std +mean,0.0,1.0)
    temp = torch.clamp((temp-mean)/std-clean,-modif_max,modif_max)
    return temp
def save_normalized_tensor_as_video(image, filename, fps=5):
    image_tensor=image[0].permute(1, 0, 2, 3)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    # 反归一化图像张量
    denormalized_tensor = (image_tensor * std) + mean
    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1)
    # 将张量转换为 NumPy 数组，并将值范围从 [0, 1] 转换为 [0, 255]
    
    denormalized_array = (denormalized_tensor.numpy() * 255 ).astype(np.uint8)
    
    # 获取图像的宽度和高度
    height, width = denormalized_array.shape[2], denormalized_array.shape[3]
    # 创建一个 VideoWriter 对象，用于将 NumPy 数组转换为 MP4 视频
    video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'FFV1'), fps, (width, height))
    # 将每个帧写入视频
    for i in range(denormalized_array.shape[0]):
        frame = denormalized_array[i].transpose(1, 2, 0)  # 将通道维度从 [C, H, W] 转换为 [H, W, C]
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 将图像从RGB转换为BGR
        
        video_writer.write(frame)
    video_writer.release()

def upload_video_without_audio(video_path, conv, vis_processor):
        msg = ""
        if isinstance(video_path, str):  # is a video path
            ext = os.path.splitext(video_path)[-1].lower()
            print(video_path)
            # image = self.vis_processor(image).unsqueeze(0).to(self.device)
            video, msg = load_video(
                video_path=video_path,
                n_frms=8,
                height=224,
                width=224,
                sampling ="uniform", return_msg = True
            )
            video = vis_processor.transform(video)
            video = video.unsqueeze(0).to('cuda')
        else:
            raise NotImplementedError
        
        conv.append_message(conv.roles[0], "<Video><ImageHere></Video> "+ msg)
        # conv.system = "You can understand the video that the user provides.  Follow the instructions carefully and explain your answers in detail."
        return video
        
        

def vllama_answer(conv,video,model,question):
    with torch.inference_mode():
        img_list=[]
        # print("video shape",video.shape)
        # print("video",video)
        image_emb, _ = model.encode_videoQformer_visual(video)
        img_list.append(image_emb)
        # print("img_list shape",img_list[0].shape)
        # print("img_list",img_list)
        conv_copy = conv.copy()
        if len(conv_copy.messages) > 0 and conv_copy.messages[-1][0] == conv_copy.roles[0] \
                    and ('</Video>' in conv_copy.messages[-1][1] or '</Image>' in conv_copy.messages[-1][1]):  # last message is image.
            conv_copy.messages[-1][1] = ' '.join([conv_copy.messages[-1][1], question])
        else:
            conv_copy.append_message(conv_copy.roles[0], question)
        conv_copy.append_message(conv_copy.roles[1], None)
        prompt = conv_copy.get_prompt()
        #print("prompt",prompt)
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to('cuda').input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)

        if conv_copy.sep =="###":
            stop_words_ids = [torch.tensor([835]).to('cuda'),
                        torch.tensor([2277, 29937]).to('cuda')]  # '###' can be encoded in two different ways.
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        else:
            stop_words_ids = [torch.tensor([2]).to('cuda')]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        # print("mixed_embs shape",mixed_embs.shape)
        # print("mixed_embs",mixed_embs)
        # # stopping_criteria
        outputs = model.llama_model.generate(
            inputs_embeds=mixed_embs,
            max_new_tokens=512,
            stopping_criteria=stopping_criteria,
            do_sample=True,
            temperature=0.2
        )
        # logits = model.llama_model.forward(inputs_embeds=mixed_embs).logits
        # print("logits shape",logits.shape)
        # prob =torch.argmax(logits,dim=-1)
        # print("prob shape",prob.shape)
        # print(model.llama_tokenizer.decode(prob[0][-1]))
        # print(prob[0][-1])
        # token = model.llama_tokenizer.convert_ids_to_tokens(29871)
        # print(token)
        output_token = outputs[0]
        #print("output token",output_token)
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        #print("output_text",output_text)
        if conv_copy.sep =="###":
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
        else:
            output_text = output_text.split(conv_copy.sep2)[0]  # remove the stop sign '###'
            output_text = output_text.split(conv_copy.roles[1]+':')[-1].strip()
        #print("output_text2",output_text)
        return output_text

    
def vllama_attack_eos(conv,video,model,args,question):
    conv_copy = conv.copy()
    if len(conv_copy.messages) > 0 and conv_copy.messages[-1][0] == conv_copy.roles[0] \
                and ('</Video>' in conv_copy.messages[-1][1] or '</Image>' in conv_copy.messages[-1][1]):  # last message is image.
        conv_copy.messages[-1][1] = ' '.join([conv_copy.messages[-1][1], question])
    else:
        conv_copy.append_message(conv_copy.roles[0], question)
    conv_copy.append_message(conv_copy.roles[1], None)
    prompt = conv_copy.get_prompt()
    prompt_segs = prompt.split('<ImageHere>')
    seg_tokens = [
        model.llama_tokenizer(
            seg, return_tensors="pt", add_special_tokens=i == 0).to('cuda').input_ids
        # only add bos to the first seg
        for i, seg in enumerate(prompt_segs)
    ]
    with torch.no_grad():
        seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]

    modif_max = args.max_modify / 255 / 0.26130258
    step = args.step / 255 / 0.26130258
    print('modif_max', modif_max)
    print('step', step)
    
    modif = torch.full_like(video,step).to('cuda')
    
    modifier = torch.nn.Parameter(modif, requires_grad=True)
    #print("modifier",modifier.dtype)
    image_clean = video.clone().detach()


    
    for iter in range(args.iter):
        torch.cuda.empty_cache()
        image_attack = image_clean + modifier
        
        img_list=[]
        image_emb, _ = model.encode_videoQformer_visual(image_attack)
        img_list.append(image_emb)
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        
        logits = model.llama_model.forward(inputs_embeds=mixed_embs).logits
        prob = torch.nn.functional.log_softmax(logits,dim=-1)
        #logprob = torch.log(prob)
        loss = -prob[0][-1,2]
        
        if loss.item() < 0.3:break
        if iter % 10 == 0:
            print(loss.item())
        loss.backward()
        grad_sign = torch.sign(modifier.grad)
        modifier.data -= grad_sign * step
        modifier.data = myclamp(modifier.data,image_clean.data,modif_max)
        modifier.grad.zero_()
        
    modifier=modifier.detach()
    image_attack = image_clean + modifier
    output_attack = vllama_answer(conv,image_attack,model,question)
    print("delta:",abs(image_attack-image_clean).mean() * 0.26130258 * 255)
    return output_attack,image_clean,image_attack,iter


def vllama_attack_eos2(conv,video,model,args,question):
    conv_copy = conv.copy()
    if len(conv_copy.messages) > 0 and conv_copy.messages[-1][0] == conv_copy.roles[0] \
                and ('</Video>' in conv_copy.messages[-1][1] or '</Image>' in conv_copy.messages[-1][1]):  # last message is image.
        conv_copy.messages[-1][1] = ' '.join([conv_copy.messages[-1][1], question])
    else:
        conv_copy.append_message(conv_copy.roles[0], question)
    conv_copy.append_message(conv_copy.roles[1], None)
    prompt = conv_copy.get_prompt()
    prompt_segs = prompt.split('<ImageHere>')
    seg_tokens = [
        model.llama_tokenizer(
            seg, return_tensors="pt", add_special_tokens=i == 0).to('cuda').input_ids
        # only add bos to the first seg
        for i, seg in enumerate(prompt_segs)
    ]
    with torch.no_grad():
        seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    
    modif_max = args.max_modify / 255 / 0.26130258
    step = args.step / 255 / 0.26130258
    print('modif_max', modif_max)
    print('step', step)
    modif = torch.full_like(video,step).to('cuda')
    modifier = torch.nn.Parameter(modif, requires_grad=True)

    image_clean = video.clone().detach()


    if conv_copy.sep =="###":
            stop_words_ids = [torch.tensor([835]).to('cuda'),
                        torch.tensor([2277, 29937]).to('cuda')]  # '###' can be encoded in two different ways.
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    else:
        stop_words_ids = [torch.tensor([2]).to('cuda')]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    for iter in range(args.iter):
        torch.cuda.empty_cache()
        image_attack = image_clean + modifier
        img_list=[]
        image_emb, _ = model.encode_videoQformer_visual(image_attack)
        img_list.append(image_emb)
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        with torch.inference_mode():
            outputs = model.llama_model.generate(
                inputs_embeds=mixed_embs,
                max_new_tokens=100,
                stopping_criteria=stopping_criteria,
                num_beams=1,
                do_sample=False,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.0,
                length_penalty=1,
                temperature=1.0,
            )
        seg_copy = copy.deepcopy(seg_tokens)
        output_token = outputs[0][1:].reshape(1,-1)
        #print(output_token)
        #print("seg_tokes",seg_copy[-1].shape)
        #print("output_token",output_token.shape)
        seg_copy[-1]=torch.cat((seg_copy[-1],output_token),dim=1)
        #print("seg_tokes",seg_copy[-1].shape)

        seg_embs_copy = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_copy]
        mixed_embs_copy = [emb for pair in zip(seg_embs_copy[:-1], img_list) for emb in pair] + [seg_embs_copy[-1]]
        mixed_embs_copy = torch.cat(mixed_embs_copy, dim=1)

        logits = model.llama_model.forward(inputs_embeds=mixed_embs_copy).logits
        prob = torch.nn.functional.log_softmax(logits,dim=-1)
        #logprob = torch.log(prob)
        loss = -prob[0,-output_token.shape[1]-1:-1,2].mean()
        
        if iter ==0 or iter % 10 == 0:
            print(loss.item())
            #print(-logprob[0,-output_token.shape[1]-1:-1,2])
        if loss.item() < 4:break
        

        loss.backward()
        
        grad_sign = torch.sign(modifier.grad)
        modifier.data -= grad_sign * step
        modifier.data = myclamp(modifier.data,image_clean.data,modif_max)
        modifier.grad.zero_()
        
    modifier=modifier.detach()
    image_attack = image_clean + modifier
    output_attack = vllama_answer(conv,image_attack,model,question)
    print("delta:",abs(image_attack-image_clean).mean() * 0.26130258 * 255)
    return output_attack,image_clean,image_attack,iter

def vllama_attack_random(conv,video,model,args,question):
    conv_copy = conv.copy()
    if len(conv_copy.messages) > 0 and conv_copy.messages[-1][0] == conv_copy.roles[0] \
                and ('</Video>' in conv_copy.messages[-1][1] or '</Image>' in conv_copy.messages[-1][1]):  # last message is image.
        conv_copy.messages[-1][1] = ' '.join([conv_copy.messages[-1][1], question])
    else:
        conv_copy.append_message(conv_copy.roles[0], question)
    conv_copy.append_message(conv_copy.roles[1], None)
    prompt = conv_copy.get_prompt()
    prompt_segs = prompt.split('<ImageHere>')
    seg_tokens = [
        model.llama_tokenizer(
            seg, return_tensors="pt", add_special_tokens=i == 0).to('cuda').input_ids
        # only add bos to the first seg
        for i, seg in enumerate(prompt_segs)
    ]
    with torch.no_grad():
        seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]

    modif_max = args.max_modify / 255 / 0.26130258
    step = args.step / 255 / 0.26130258
    print('modif_max', modif_max)
    print('step', step)
    modif = torch.full_like(video,step).to('cuda')
    modifier = torch.nn.Parameter(modif, requires_grad=True)

    image_clean = video.clone().detach()
    mse_loss = torch.nn.MSELoss()
    

    img_list=[]
    with torch.no_grad():
        image_clean_emb,_ = model.encode_videoQformer_visual(image_clean)
        img_list.append(image_clean_emb)
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        llm_clean_feature = model.llama_model.attack(inputs_embeds=mixed_embs)
    for iter in range(args.iter):
        torch.cuda.empty_cache()
        image_attack = image_clean + modifier
        
        img_list=[]
        image_emb, _ = model.encode_videoQformer_visual(image_attack)
        img_list.append(image_emb)
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        llm_feature = model.llama_model.attack(inputs_embeds=mixed_embs)
        loss1 = mse_loss(image_emb,image_clean_emb)
        loss2 = mse_loss(llm_feature, llm_clean_feature)
        loss = -args.weight_clip * loss1 - args.weight_llm * loss2
        loss.backward()
        if iter % 10 == 0:
            print(loss.item())
        grad_sign = torch.sign(modifier.grad)
        modifier.data -= grad_sign * step
        modifier.data = myclamp(modifier.data,image_clean.data,modif_max)
        modifier.grad.zero_()
        
        
    modifier=modifier.detach()
    image_attack = image_clean + modifier
    output_attack = vllama_answer(conv,image_attack,model,question)
    print("delta:",abs(image_attack-image_clean).mean() * 0.26130258 * 255)
    return output_attack,image_clean,image_attack,iter

def vllama_attack_train(conv,video,model,args,question):
    conv_copy = conv.copy()
    if len(conv_copy.messages) > 0 and conv_copy.messages[-1][0] == conv_copy.roles[0] \
                and ('</Video>' in conv_copy.messages[-1][1] or '</Image>' in conv_copy.messages[-1][1]):  # last message is image.
        conv_copy.messages[-1][1] = ' '.join([conv_copy.messages[-1][1], question])
    else:
        conv_copy.append_message(conv_copy.roles[0], question)
    conv_copy.append_message(conv_copy.roles[1], None)
    prompt = conv_copy.get_prompt()
    prompt_segs = prompt.split('<ImageHere>')
    seg_tokens = [
        model.llama_tokenizer(
            seg, return_tensors="pt", add_special_tokens=i == 0).to('cuda').input_ids
        # only add bos to the first seg
        for i, seg in enumerate(prompt_segs)
    ]
    with torch.no_grad():
        seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    
    modif_max = args.max_modify / 255 / 0.26130258
    step = args.step / 255 / 0.26130258
    print('modif_max', modif_max)
    print('step', step)
    modif = torch.full_like(video,step).to('cuda')
    modifier = torch.nn.Parameter(modif, requires_grad=True)

    image_clean = video.clone().detach()
    

    if conv_copy.sep =="###":
            stop_words_ids = [torch.tensor([835]).to('cuda'),
                        torch.tensor([2277, 29937]).to('cuda')]  # '###' can be encoded in two different ways.
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    else:
        stop_words_ids = [torch.tensor([2]).to('cuda')]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    img_list=[]
    with torch.no_grad():
        image_emb, _ = model.encode_videoQformer_visual(image_clean)
    img_list.append(image_emb)
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    with torch.inference_mode():
        outputs = model.llama_model.generate(
            inputs_embeds=mixed_embs,
            max_new_tokens=100,
            stopping_criteria=stopping_criteria,
            num_beams=1,
            do_sample=False,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=1.0,
        )
    output_token = outputs[0][1:].reshape(1,-1)

    for iter in range(args.iter):
        torch.cuda.empty_cache()
        image_attack = image_clean + modifier
        img_list=[]
        image_emb, _ = model.encode_videoQformer_visual(image_attack)
        img_list.append(image_emb)
        
    
        seg_copy = copy.deepcopy(seg_tokens)
        seg_copy[-1]=torch.cat((seg_copy[-1],output_token),dim=1)
        #print("seg_tokes",seg_copy[-1].shape)

        seg_embs_copy = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_copy]
        mixed_embs_copy = [emb for pair in zip(seg_embs_copy[:-1], img_list) for emb in pair] + [seg_embs_copy[-1]]
        mixed_embs_copy = torch.cat(mixed_embs_copy, dim=1)
        #print("mixed",mixed_embs_copy.shape)
        labels = torch.full((1,mixed_embs_copy.shape[1]), -100).to('cuda')
        for i in range(1,1+output_token.shape[1]):
            labels[0,-i] = output_token[0,-i]
        loss = -model.llama_model.forward(inputs_embeds=mixed_embs_copy,labels=labels).loss
        
        if iter ==0 or iter % 10 == 0:
            print(loss.item())
        

        loss.backward()
        
        grad_sign = torch.sign(modifier.grad)
        modifier.data -= grad_sign * step
        modifier.data = myclamp(modifier.data,image_clean.data,modif_max)
        modifier.grad.zero_()
        
    modifier=modifier.detach()
    image_attack = image_clean + modifier
    output_attack = vllama_answer(conv,image_attack,model,question)
    print("delta:",abs(image_attack-image_clean).mean() * 0.26130258 * 255)
    return output_attack,image_clean,image_attack,iter

def attack_random(conv,video,model,args,question):
    
    step = args.step / 255 / 0.26130258
    modif_max=args.max_modify / 255 / 0.26130258
    modif = (torch.rand_like(video) -0.5) * 2 * args.max_modify * step
    #print(modif)
    modif = modif.to('cuda').detach()
    


    image_clean = video.clone().detach()
    
    modif.data = myclamp(modif.data,image_clean.data,modif_max)
    image_attack = image_clean + modif
    output_attack = vllama_answer(conv,image_attack,model,question)
    print("delta:",abs(image_attack-image_clean).mean() * 0.26130258 * 255)
    return output_attack,image_clean,image_attack,1

def run_inference(args):
    print('Initializing Chat')
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()
    model.requires_grad_(False)

    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    print('Initialization Finished')
    # Load the ground truth file
    # with open(args.gt_file) as file:
    #     dataset = json.load(file)
    
    with open(args.gt_file, 'r') as csvfile:
        dataset = list(csv.DictReader(csvfile))

    save_path = f'clip{args.weight_clip}_llm{args.weight_llm}_{args.type}_iter{args.iter}_mod{args.max_modify}_step{args.step}'
    # Create the output directory if it doesn't exist
    if not os.path.exists(os.path.join(args.output_dir, save_path)):
        os.makedirs(os.path.join(args.output_dir, save_path))
    sum_clean,sum_attack=0,0
    for sample in dataset:
        torch.cuda.empty_cache()
        if args.model_type == 'vicuna':
            chat_state = default_conversation.copy()
        else:
            chat_state = conv_llava_llama_2.copy()
        chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        video_path = os.path.join(args.video_dir, f"{sample['video_name']}")
        video = upload_video_without_audio(video_path, chat_state, vis_processor)
        question = "What is this video about?"
        print("question:",question)
        # clean_text = vllama_answer(chat_state, video, model,question)
        # print("clean_text:",clean_text)
        if args.type == 'eos':
            attack_text,image_clean,image_attack,epoch = vllama_attack_eos(conv = chat_state, video = video, model = model, args=args, question=question)
        elif args.type == 'eos2':
            attack_text,image_clean,image_attack,epoch = vllama_attack_eos2(conv = chat_state, video = video, model = model, args=args, question=question)
        elif args.type == 'random':
            attack_text,image_clean,image_attack,epoch = vllama_attack_random(conv = chat_state, video = video, model = model, args=args, question=question)
        elif args.type == 'train':
            attack_text,image_clean,image_attack,epoch = vllama_attack_train(conv = chat_state, video = video, model = model, args=args, question=question)
        elif args.type == 'attack_random':
            attack_text,image_clean,image_attack,epoch = attack_random(conv = chat_state, video = video, model = model, args=args, question=question)
        
        with open(os.path.join(args.output_dir, save_path, f"attack.csv"), 'a', encoding='utf-8',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([sample['video_name'], attack_text])
        #save_normalized_tensor_as_video(image_clean.cpu().detach(), os.path.join(args.output_dir, save_path, f"{sample['video_name'].replace('.avi','')}_clean_video.avi"))
        save_normalized_tensor_as_video(image_attack.cpu().detach(), os.path.join(args.output_dir, save_path, f"{sample['video_name'].replace('.avi','')}_attack_video.avi"))


if __name__ == "__main__":
    args = parse_args()
    myseed=1
    random.seed(myseed)
    os.environ["PYTHONHASHSEED"] = str(myseed)
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    torch.backends.cudnn.benchmark = False   
    torch.backends.cudnn.deterministic = True
    run_inference(args)
