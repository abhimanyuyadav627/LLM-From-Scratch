import torch
import tiktoken
import torch.nn as nn
import os
from dotenv import load_dotenv
from gpt_architecture.custom_gpt import GPTModel, GPT_CONFIG_124M
from pretrain.data_loader import create_dataloader
from pretrain.text_generation import TextGenerator
from pretrain.load_pretrained_weights import get_gpt_with_openai_gpt2_weights

# Load environment variables from .env file
load_dotenv()

def calculate_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())
    return loss

def calculate_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_dataloader, validation_dataloader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss_loader(train_dataloader, model, device, num_batches=eval_iter)
        val_loss = calculate_loss_loader(validation_dataloader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context, temperature, top_k):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = TextGenerator.text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = TextGenerator.generate_text(model=model, idx=encoded,max_new_tokens=25, context_size=context_size, temperature=temperature,top_k = top_k)
        # token_ids = TextGenerator.generate_text_simple(model=model, idx=encoded,max_new_tokens=50, context_size=context_size)
        decoded_text = TextGenerator.token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " ")) # Compact print format
    model.train()

def training_loop(model,train_dataloader,validation_dataloader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses,val_losses,track_tokens_seen = [],[],[]
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs): #one epoch is one complete pass over the training set.
        model.train()
        for input_batch, target_batch in train_dataloader:
           optimizer.zero_grad() #1. resetting loss gradient from the previous step.
           loss = calculate_loss_batch(input_batch=input_batch,target_batch=target_batch,model=model,device=device) #2. calculating loss on current batch.
           loss.backward() #3. backward pass to calculate loss gradients using pytorch autograd engine.
           optimizer.step() #4. updating model weights using computed loss gradients
           tokens_seen += input_batch.numel()
           global_step += 1
           if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                model, train_dataloader, validation_dataloader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                f"Train loss {train_loss:.3f}, "
                f"Val loss {val_loss:.3f}"
                )
        generate_and_print_sample( model, tokenizer, device, start_context,temperature=float(os.getenv('TEMPERATURE')), top_k = int(os.getenv('TOP_K'))) #REMINDER: Adjust temprature over here
    return train_losses,val_losses,track_tokens_seen


if __name__ == "__main__":
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    train_ratio = 0.90
    split_idx = int(train_ratio * len(raw_text))
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]
    train_dataloader = create_dataloader(train_data, max_length=GPT_CONFIG_124M["context_length"], stride=GPT_CONFIG_124M["context_length"], batch_size=2 , shuffle=True)
    validation_dataloader = create_dataloader(val_data, max_length=GPT_CONFIG_124M["context_length"], stride=GPT_CONFIG_124M["context_length"], batch_size=2 , shuffle=False)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    # model = GPTModel(GPT_CONFIG_124M)
    model = get_gpt_with_openai_gpt2_weights()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(os.getenv('LEARNING_RATE')), weight_decay=float(os.getenv('WEIGHT_DECAY')))
    num_epochs = int(os.getenv('EPOCHS'))
    train_losses, val_losses, tokens_seen = training_loop(model, train_dataloader, validation_dataloader, optimizer, 
                                                               device,num_epochs=num_epochs, eval_freq=int(os.getenv('EVAL_FREQ')), eval_iter=int(os.getenv('EVAL_ITER')),start_context=os.getenv('START_CONTEXT'), tokenizer=tokenizer)

    #code to save the pretrained model. (saving the optimizer states so that we can continue pretraining LLM.)
    torch.save({"model_state_dict": model.state_dict(),"optimizer_state_dict": optimizer.state_dict(),},"saved_models/model_and_optimizer.pth")
    print("INFO: Pretrained model saved successfully.")