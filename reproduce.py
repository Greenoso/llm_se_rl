import json
import os
import sys
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig



task='autodl-tmp/BIG-Bench-Hard/bbh/penguins_in_a_table.json'










default_config = TRLConfig(
    train=TrainConfig(
        seq_length=512,
        epochs=100,
        total_steps=6000,
        # gradient accummulation=4 via deepspeed
        batch_size=16,
        checkpoint_interval=500,
        eval_interval=50,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        save_best=False,
        tracker="wandb",
        checkpoint_dir='/root/autodl-tmp/msc_ml/t5_large_checkpoints'
    ),
    model=ModelConfig(
        model_path="/root/autodl-tmp/flan-t5-large",
        num_layers_unfrozen=2,
        model_arch_type="seq2seq",
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="/root/autodl-tmp/flan-t5-large",
        padding_side="right",
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 1.0e-4,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 100000,
            "eta_min": 1.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        ### reduce rollouts due to small dataset
        num_rollouts=128,
        chunk_size=12,
        ppo_epochs=4,
        init_kl_coef=0.1,
        target=6,
        horizon=1000,
        gamma=0.99,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 128,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "eos_token_id": T5Tokenizer.from_pretrained("/root/autodl-tmp/flan-t5-large").eos_token_id,
            "temperature": 1.0,
        },
    ),
)


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)


    
    #########################################b
    
    ### reward_se
    def reward_se( prompts: List[str], outputs: List[str], **kwargs) -> List[float]:



        rewards = []
        for q, a in zip(prompts, outputs):
            feedback_prompt = f'Is the answer to the question correct? The question is: {q}. The answer is: {a}.'
            feedback = se_generator(feedback_prompt)[0]['generated_text']  # Assuming 'model' is your trained T5 model
            feedback = feedback.lower().strip()
            print(feedback)
            reward = 0.0 
            if 'yes' in feedback:
                reward = 1.0 
                
            elif 'no' in feedback:
                reward = -1.0

            rewards.append(reward)
        return rewards
    
    
    
    
    
    ### metric_se

    
    def metric_se(samples: List[str], prompts: List[str], outputs: List[str]) -> Dict[str, List[float]]:
        match=[]
        
        for i,prompt in enumerate(prompts):

            index = prompt_all_new.index(prompt)
            if outputs[i].lower().strip()==answer_all[index].lower().strip():
                is_correct=1.0
            else:
                is_correct=0.0
                
            match.append(is_correct)

        return {"Answer Matching": match}
    
    ###########################################e
    
    
    
    ############################b
    # Load the model
    model_se = T5ForConditionalGeneration.from_pretrained("/root/autodl-tmp/flan-t5-large")

    # Load the tokenizer
    tokenizer_se = AutoTokenizer.from_pretrained("/root/autodl-tmp/flan-t5-large")

    # Create the pipeline
    se_generator = pipeline("text2text-generation", model=model_se, tokenizer=tokenizer_se,
                        do_sample= False,
                        max_length=64,
                        eos_token_id= tokenizer_se.eos_token_id,
        device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,)
    #############################e


    
    

    
    #########################b

    ds = load_dataset("json", data_files="/root/autodl-tmp/BIG-Bench-Hard/bbh/navigate.json",field="examples")['train']
   
    
    answer_all=ds['target']
    
    prompt_all=ds['input']
    # fix the train test split
    train_test_split_id=round(len(answer_all)*0.8)
    
    # ds_split=ds.train_test_split(test_size=0.2)
    #prompt_train=ds_split['train']['input']
    #prompt_test=ds_split['test']['input']
    
    
    prompt_train=prompt_all[:train_test_split_id]
    prompt_test=prompt_all[train_test_split_id:]

    
    prompt_all_cot= ['[{}] Let’ s think step by step.'.format(prompt) for prompt in prompt_all]
    prompt_test_cot= ['[{}] Let’ s think step by step.'.format(prompt) for prompt in prompt_test]
    prompt_train_cot= ['[{}] Let’ s think step by step.'.format(prompt) for prompt in prompt_train]    

    ##########################e


    trlx.train(
        prompts=prompt_train_cot,
        eval_prompts=prompt_test_cot,
        reward_fn=reward_se,
        #metric_fn=metric_se,
        config=config,
    )


    

    
    
if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    
    main(hparams)