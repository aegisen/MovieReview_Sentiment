import torch
from torch import nn
import pandas as pd 
import re
import matplotlib.pyplot as plt
import numpy as np
import json
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,TensorDataset
import pandas as pd
from DataLoader import MovieDataset
from LSTM import LSTMModel
from GloveEmbed import _get_embedding
import time
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# save checkpoint
def _save_checkpoint(ckp_path, model, epoches, global_step, optimizer):
    checkpoint = {'epoch': epoches,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, ckp_path)

def _compute_accuracy(y_pred, y_batch):
  # compute accuracy
  accy = sum(y_pred == y_batch) / len(y_batch)
  return accy

def adjust_learning_rate(learning_rate, optimizer, epoch):
  """Sets the learning rate to the initial LR decayed by 1/10 every args.lr epochs"""
  lr = learning_rate
  if (epoch > 5):
    lr = 0.001
  if (epoch >= 10):
    lr = 0.0001
  if (epoch > 20):
    lr = 0.00001
  for param_group in optimizer.param_groups:
      param_group['lr'] = lr

def main():
    gpu_id = 0
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda", gpu_id)
    else:
        device = torch.device('cpu')
    
    ## alternative method
    # torch.cuda.set_device(device=0) ## choose gpu number
    print('device: ', device)

    # set params
    mode = 'train'
    Batch_size = 350
    n_layers = 2 ## choose 1-3 layers

    # input seq length acc. to data pre-processing
    input_len = 150

    # word embedding length
    embedding_dim = 300

    # LSTM hidden dim
    hidden_dim = 100
    # binary cross entropy
    output_size = 1
    num_epoches = 50

    learning_rate = 0.001

    # gradient clipping
    clip = 5
    load_cpt = False #True
    ckp_path = 'checkpoint/name.pt'
    global_step = 0 if not load_cpt else checkpoint['global_step']  #temp
    # embedding_matrix = None
    ## use pre-train Glove embedding
    pretrain = True #use glove embedding

    # uncomment to pick embedding
    #glove_file = "glove.6B.50d.txt"
    #glove_file = "glove.6B.100d.txt" 
    #glove_file = "glove.6B.200d.txt"
    glove_file = "glove.6B.300d.txt"
    
    # load training and test data
    training_set = MovieDataset('new_train_data.csv')

    training_generator = DataLoader(training_set, batch_size=Batch_size,\
                                    shuffle=True,num_workers=1)
    test_set = MovieDataset('new_test_data.csv')
    #print(test_set.df.head())
    test_generator = DataLoader(test_set, batch_size=Batch_size,\
                                shuffle=False,num_workers=1)


    # read tokens and load pre-train embedding
    with open('tokens2index.json', 'r') as f:
        tokens2index = json.load(f)
    vocab_size = len(tokens2index)

    if pretrain:
        print('***** load glove embedding now...****')
        embedding_matrix = _get_embedding(glove_file,tokens2index,embedding_dim)
    else:
        embedding_matrix = None

    # import and load LSTM model
    model = LSTMModel(input_len = input_len,output_size = output_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim, n_layers = n_layers, pretrain = pretrain, embedding_matrix = embedding_matrix, vocab_size = vocab_size)
    model.to(device)

    # setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  

    # and loss function
    loss_fun = nn.BCELoss()
    
    # load checkpoint
    if load_cpt:
        print("*"*10+'loading checkpoint'+'*'*10)

        checkpoint = torch.load(ckp_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoches = checkpoint["epoch"]
        


    # train model
    print('*'*89)
    print('start model training now')
    print('*'*89)
    if mode == 'train':
        model.train()
        accuracies = []
        losses = []
        losses_iter = []
        losses_dict = {}
        losses_dict_iter = {}
        for epoches in range(num_epoches):
            cur_loss = 0.0
            #Adjust learning rate
            adjust_learning_rate(learning_rate, optimizer, epoches)
            for x_batch, y_labels in training_generator:
                
                global_step += 1
                x_batch, y_labels = x_batch.to(device), y_labels.to(device)
                # get prediction result
                y_out = model(x_batch)
                y_pred = torch.round(y_out)

                # get loss
                loss = loss_fun(y_out, y_labels)

                # back propogation
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                # get accuracy
                acc = _compute_accuracy(y_pred, y_labels)
                accuracies.append(acc)

                # keep track of loss
                cur_loss += loss.item()

                # loss for iter
                losses_dict_iter[global_step] = loss.item()
                losses_iter.append(loss.item())
                
            # save checkpoint
            def _save_checkpoint(ckp_path, model, epoches, global_step, optimizer):
              checkpoint = {"epoch":epoches, "global_step":global_step, "model_state_dict":model.state_dict(), "optimizer_state_dict":optimizer.state_dict()}
              torch.save(checkpoint, ckp_path)
            #print("**** save checkpoint ****")
            ckp_path = "checkpoint/step_{}.pt".format(global_step)
            #_save_checkpoint(ckp_path, model, epoches, global_step, optimizer)

            #Calculate loss for epoch
            epoch_loss = cur_loss/len(training_generator)
            #Enter into dictionary to plot later
            losses_dict[epoches] = epoch_loss
            losses.append(epoch_loss)
            #print("Loss by epoch (dict): ", losses_dict)

            


        #Epoch: sort by key
#        losses_data = sorted(losses_dict.items()) 
#        #unpack list of pairs into two tuples
#        x, y = zip(*losses_data)
#        plt.xlabel("Epoch")
#        plt.ylabel("Loss")
        #Plot loss over epoch
#        plt.plot(x, y)
#        plt.show()

        #Iter: sort by key
        losses_data = sorted(losses_dict_iter.items()) 
        #unpack list of pairs into two tuples
        x, y = zip(*losses_data)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        #Plot loss over Iteration
        plt.plot(x, y)
        plt.show()
            
    
    print("----model testing now----")
    model.eval()
    with torch.no_grad():
      accuracies = []
      predictions = []
      true_vals = []
      for x_batch, y_labels in test_generator:
        x_batch, y_labels = x_batch.to(device), y_labels.to(device)
        y_out = model(x_batch)
        y_pred = torch.round(y_out)

        predictions.extend(y_pred.tolist())
        true_vals.extend(y_labels.tolist())
        #Get Accuracy
        accy = _compute_accuracy(y_pred, y_labels)
        #print("Test accuracy is: ", accy)
			
        accuracies.append(accy)
        #return accuracies

      print("Test accuracy is: ", (sum(accuracies)/len(accuracies)))

    print('Classification Report:')
    print(classification_report(true_vals, predictions, labels=[1,0], digits=4))
    
    cm = confusion_matrix(true_vals, predictions, labels=[1,0])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['FALSE', 'TRUE'])
    ax.yaxis.set_ticklabels(['FALSE', 'TRUE'])


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print("running time: ", (time_end - time_start)/60.0, "mins")
    
