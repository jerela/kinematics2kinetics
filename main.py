import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import CustomTimeSeriesDataset
from networks import KineticsLSTM
from visualization import plotter

torch.manual_seed(1)


def train(model, dataloader, n_epochs = 100, lr = 0.01, threshold = 0.01):
    
    # select optimizer and learning rate
    optimizer = Adam(model.parameters(), lr=lr)
    
    # keep track of previous total loss so we know to break out of the epoch loop if total loss doesn't change between epochs
    previous_total_loss = 0.0
    
    # iterate through epochs, n_epochs is the maximum number of epochs allowed
    for epoch in range(n_epochs):
        total_loss = 0.0
        
        for i_input, i_target in dataloader:
            i_output = model(i_input).permute(0,2,1)
            
            loss = ((i_output - i_target)**2).mean()
            
            loss.backward()
            
            total_loss += loss
            
        
        #if total_loss < 50:
        if abs(total_loss-previous_total_loss) < threshold:
            print(f'Steps taken: {str(epoch)}')
            break
        
        previous_total_loss = total_loss
        
        optimizer.step()
        optimizer.zero_grad()
        
        print(f'Step: {str(epoch+1)}, final loss: {str(total_loss)}')
    


def main():
    
    
    dataset = CustomTimeSeriesDataset('C:/Users/lavik/OneDrive/Documents/Lencioni/processed_data.csv')
    
    dataloader = DataLoader(dataset, shuffle=True)
    
    sample_input, sample_target = dataset[0]
    
    sample_input = sample_input.unsqueeze(0)
    sample_target = sample_target.unsqueeze(0).permute(0,2,1)
    
    n_inputs, n_targets = dataset.get_num_features()
    model = KineticsLSTM(n_inputs,n_targets)
    
    output_untrained = model(sample_input).detach()
    
    train(model=model, dataloader=dataloader, n_epochs=10000, threshold=0.05)

    output_trained = model(sample_input).detach()

    loss_untrained = ((sample_target-output_untrained)**2).mean()
    loss_trained = ((sample_target-output_trained)**2).mean()
    print(f'Loss untrained: {str(loss_untrained)}')
    print(f'Loss trained: {str(loss_trained)}')
    

    plottable_data = (sample_target.detach().squeeze(0), output_untrained.squeeze(0), output_trained.squeeze(0))
    plottable_titles = ('target', 'untrained prediction', 'trained prediction')
    plotter(plottable_data, plottable_titles)
    

if __name__ == "__main__":
    main()




