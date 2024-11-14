import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch import torch, nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
        # print(X,y)
    return np.array(X), np.array(y)

class LSTM(nn.Module):
    
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_classes = num_classes # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2) # lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) # fully connected 
        self.fc_2 = nn.Linear(128, num_classes) # fully connected last layer
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # cell state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # (input, hidden, and internal state)
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) # first dense
        out = self.relu(out) # relu
        out = self.fc_2(out) # final output
        return out
    
def training_loop(n_epochs, lstm, optimiser, loss_fn, X_train, y_train,
                  X_test, y_test):
    for epoch in range(n_epochs):
        lstm.train()
        outputs = lstm.forward(X_train) # forward pass
        optimiser.zero_grad() # calculate the gradient, manually setting to 0
        # obtain the loss function
        loss = loss_fn(outputs, y_train)
        loss.backward() # calculates the loss of the loss function
        optimiser.step() # improve from loss, i.e backprop
        # test loss
        lstm.eval()
        test_preds = lstm(X_test)
        test_loss = loss_fn(test_preds, y_test)
        if epoch % 100 == 0:
            print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, 
                                                                      loss.item(), 
                                                                      test_loss.item())) 
        if epoch == n_epochs-1:
            train_loss_save.append(loss.item())
            test_loss_save.append(test_loss.item())

df = pd.read_csv('processed_data/revised_processed_data.csv')
naics = df['NAICS'].unique()
naics_titles = np.array(['Agriculture, Forestry, Fishing and Hunting',
       'Mining, Quarrying, and Oil and Gas Extraction', 'Utilities',
       'Construction', 'Manufacturing', 'Wholesale Trade', 'Retail Trade',
       'Transportation and Warehousing', 'Information',
       'Finance and Insurance', 'Real Estate and Rental and Leasing',
       'Professional, Scientific, and Technical Services',
       'Management of Companies and Enterprises',
       'Administrative and Support and Waste Management and Remediation Services',
       'Educational Services', 'Health Care and Social Assistance',
       'Arts, Entertainment, and Recreation',
       'Accommodation and Food Services',
       'Other Services (except Public Administration)',
       'Federal, State, and Local Government, excluding State and Local Government Schools and Hospitals and the U.S. Postal Service (OEWS Designation)'],
      dtype=object)
pred_len = 2
feed_days = 5
train_loss_save = []
test_loss_save = []
for i,code in enumerate(naics):
    all0 = df[df['NAICS'] == code]
    X = all0.drop(columns=['TOT_EMP','NAICS','YEAR'])
    y = all0['TOT_EMP'].values
    mm = MinMaxScaler()
    ss = StandardScaler()

    X_trans = ss.fit_transform(X)
    y_trans = mm.fit_transform(y.reshape(-1, 1)) 
    X_ss, y_mm = split_sequences(X_trans, y_trans, feed_days, pred_len)
    X_train = X_ss[:-pred_len]
    X_test = X_ss[-pred_len:]

    y_train = y_mm[:-pred_len]
    y_test = y_mm[-pred_len:]
    X_train_tensors = torch.Tensor(X_train)
    X_test_tensors = torch.Tensor(X_test)

    y_train_tensors = torch.Tensor(y_train)
    y_test_tensors = torch.Tensor(y_test)
    X_train_tensors_final = torch.reshape(X_train_tensors,   
                                      (X_train_tensors.shape[0], feed_days, 
                                       X_train_tensors.shape[2]))
    X_test_tensors_final = torch.reshape(X_test_tensors,  
                                        (X_test_tensors.shape[0], feed_days, 
                                        X_test_tensors.shape[2])) 
    n_epochs = 300 # 300 epochs
    learning_rate = 0.001 # 0.001 lr

    input_size = 10 # number of features
    hidden_size = 2 # number of features in hidden state
    num_layers = 1 # number of stacked lstm layers

    num_classes = pred_len # number of output classes 

    lstm = LSTM(num_classes, 
                input_size, 
                hidden_size, 
                num_layers)
    loss_fn = torch.nn.MSELoss()    # mean-squared error for regression
    optimiser = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    training_loop(n_epochs=n_epochs,
              lstm=lstm,
              optimiser=optimiser,
              loss_fn=loss_fn,
              X_train=X_train_tensors_final,
              y_train=y_train_tensors,
              X_test=X_test_tensors_final,
              y_test=y_test_tensors)
    test_predict = lstm(X_test_tensors_final[-1].unsqueeze(0)) # get the last sample
    test_predict = test_predict.detach().numpy()
    test_predict = mm.inverse_transform(test_predict)
    test_predict = test_predict[0].tolist()

    test_target = y_test_tensors[-1].detach().numpy() # last sample again
    test_target = mm.inverse_transform(test_target.reshape(1, -1))
    test_target = test_target[0].tolist()

    #Plotting
    plt.figure(figsize=(10,6)) #plotting
    a = [x for x in range(len(y))]
    plt.plot(a, y, label='Actual data')
    c = [x for x in range(len(y)-pred_len-1, len(y))]
    pred_trend = np.insert(test_predict,0,y[-pred_len-1])
    plt.plot(c, pred_trend, label='Prediction for 2022-2023 data')
    plt.xticks(all0['YEAR'].values-2005)
    plt.axvline(x=len(y)-pred_len-1, c='r', linestyle='--')
    plt.xlabel("Years Since 2005")
    plt.ylabel("Total Employment")
    plt.title(f"Total Employment across {naics_titles[i]} over time")
    plt.legend()
    plt.savefig(f"lstm_plots2/predict_naics_{code}.png", dpi=300)
    plt.clf()
print(f'Training loss: {train_loss_save}')
print(f'Test loss: {test_loss_save}')