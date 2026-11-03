import json,re,time
class LocalQwen:
    def __init__(self,path):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        self.torch=torch;assert torch.cuda.is_available()
        torch.set_num_threads(4);torch.manual_seed(20261031);start=time.time()
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,
          device_map='auto',max_memory={0:'27GiB','cpu':'8GiB'},attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
        self.load_seconds=time.time()-start
        print('LOCAL_QWEN_LOADED',round(self.load_seconds,2),flush=True)
    def generate(self,messages,tools):
        torch=self.torch
        prompt=self.tokenizer.apply_chat_template(messages,tools=tools,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        x=self.tokenizer(prompt,return_tensors='pt').to(self.model.device)
        n=x.input_ids.shape[1]
        if n>24576:raise RuntimeError('T5 context budget exceeded: '+str(n))
        torch.cuda.synchronize();start=time.time()
        with torch.inference_mode():y=self.model.generate(**x,max_new_tokens=768,do_sample=False,pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize();elapsed=time.time()-start
        raw=self.tokenizer.decode(y[0,n:],skip_special_tokens=False)
        calls=[];parse_errors=[]
        for text in re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',raw,re.S):
            try:
                call=json.loads(text)
                if isinstance(call.get('arguments'),str):call['arguments']=json.loads(call['arguments'])
                calls.append(call)
            except (ValueError,TypeError) as exc:parse_errors.append(str(exc))
        return raw,calls,{'input_tokens':n,'output_tokens':len(y[0,n:]),'generation_seconds':elapsed,'parse_errors':parse_errors}
