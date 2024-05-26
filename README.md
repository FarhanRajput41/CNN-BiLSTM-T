# CNN-BiLSTM-T
!pip install -q torchviz
from google.colab import drive
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
from torchviz import make_dot
df = pd.read_csv('/content/drive/MyDrive/Data/Satalite Hourly Data/Bahawalpur.csv')
df.head()
	YEAR	MO	DY	HR	ALLSKY_SFC_SW_DWN	QV2M	RH2M	PRECTOTCORR	T2M	WD10M	WS10M
0	2019	1	1	5	0.00	3.36	52.25	0.0	7.48	118.06	2.28
1	2019	1	1	6	0.00	3.30	54.94	0.0	6.51	132.83	2.62
2	2019	1	1	7	24.47	3.30	54.19	0.0	6.62	149.15	2.71
3	2019	1	1	8	159.88	3.11	41.75	0.0	9.69	160.76	3.53
4	2019	1	1	9	326.04	2.93	34.31	0.0	11.76	165.63	3.94
df.to_csv('data_before_FE')
df.drop(['YEAR','DY'],axis=1,inplace=True)
df.info()

df.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 26304 entries, 0 to 26303
Data columns (total 9 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   MO                 26304 non-null  int64  
 1   HR                 26304 non-null  int64  
 2   ALLSKY_SFC_SW_DWN  26304 non-null  float64
 3   QV2M               26304 non-null  float64
 4   RH2M               26304 non-null  float64
 5   PRECTOTCORR        26304 non-null  float64
 6   T2M                26304 non-null  float64
 7   WD10M              26304 non-null  float64
 8   WS10M              26304 non-null  float64
dtypes: float64(7), int64(2)
memory usage: 1.8 MB
df.isnull().sum()
MO                   0
HR                   0
ALLSKY_SFC_SW_DWN    0
QV2M                 0
RH2M                 0
PRECTOTCORR          0
T2M                  0
WD10M                0
WS10M                0
dtype: int64
df.describe().to_csv('data_stats')
df.describe()
	MO	HR	ALLSKY_SFC_SW_DWN	QV2M	RH2M	PRECTOTCORR	T2M	WD10M	WS10M
count	26304.000000	26304.000000	26304.000000	26304.000000	26304.000000	26304.000000	26304.000000	26304.000000	26304.00000
mean	6.521898	11.500000	216.464280	9.026087	39.790581	0.040945	27.029361	162.401334	2.91875
std	3.449052	6.922318	290.228687	4.951921	19.573369	0.530984	10.419445	91.406757	1.51315
min	1.000000	0.000000	0.000000	1.160000	3.190000	0.000000	1.820000	0.000000	0.01000
25%	4.000000	5.750000	0.000000	4.640000	24.940000	0.000000	18.610000	86.420000	1.89000
50%	7.000000	11.500000	11.305000	7.690000	36.560000	0.000000	28.240000	172.305000	2.70000
75%	10.000000	17.250000	432.362500	13.325000	51.880000	0.000000	35.420000	222.682500	3.64000
max	12.000000	23.000000	1007.110000	23.990000	100.000000	41.470000	49.560000	359.940000	11.80000
**#Continuous and discrete features transformation and standardization.**
hour_column = df['HR']
month_column = df['MO']
continuous_features = df.drop(['HR', 'MO'], axis=1)
cf = list(continuous_features.columns)

hour_mapping = {0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 3, 19: 3, 20: 3, 21: 3, 22: 3, 23: 3}
hour_column = hour_column.map(hour_mapping)

# Convert months to 2 discrete values: winter , summer
month_mapping = {1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 1}
month_column = month_column.map(month_mapping)

# Step 3: Standardize the remaining continuous columns
scaler = StandardScaler()
continuous_features = pd.DataFrame(scaler.fit_transform(continuous_features))
continuous_features.columns = cf
# Step 4: Concatenate the transformed columns back together
data = pd.concat([hour_column, month_column, pd.DataFrame(continuous_features)], axis=1)
target_col = data.pop('ALLSKY_SFC_SW_DWN')
data['target'] = target_col
data.reset_index(inplace=True)
data.drop('index',axis=1,inplace=True)
data.head()
	HR	MO	QV2M	RH2M	PRECTOTCORR	T2M	WD10M	WS10M	target
0	3	1	-1.144242	0.636562	-0.077112	-1.876274	-0.485108	-0.422141	-0.745855
1	1	1	-1.156359	0.773996	-0.077112	-1.969371	-0.323520	-0.197440	-0.745855
2	1	1	-1.156359	0.735678	-0.077112	-1.958813	-0.144974	-0.137960	-0.661540
3	1	1	-1.194728	0.100108	-0.077112	-1.664166	-0.017957	0.403966	-0.194968
4	1	1	-1.231078	-0.280007	-0.077112	-1.465496	0.035323	0.674929	0.377557
data.to_csv('data_after_FE')
**Model Building
Input -> CNN -> BILSTM -> Transformer Encoder -> Linear -> Output**
# Step 1: Define the CNN-LSTM Transformer model
# class CNNLSTMTransformer(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(CNNLSTMTransformer, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv1d(input_size, 32, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#         )
#         self.lstm = nn.LSTM(32, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         x = self.cnn(x)
#         x, _ = self.lstm(x)
#         x = self.fc(x[:, -1, :])
#         return x

############################################################################################
class CNNBiLSTMTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNNBiLSTMTransformer, self).__init__()

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # LSTM layers
        self.bilstm = nn.LSTM(32, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Transformer layers
        self.transformer = nn.TransformerEncoderLayer(d_model=2*hidden_size, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer, num_layers=2)

        # Fully connected layer
        self.fc = nn.Linear(2*hidden_size, num_classes)

    def forward(self, x):
        # CNN
        x = self.cnn(x)

        # Reshape for LSTM
        x = x.permute(0, 2, 1)

        # LSTM
        x, _ = self.bilstm(x)

        # Transformer input preparation
        x = x.permute(1, 0, 2)  # Reshape for transformer

        # Transformer
        x = self.transformer_encoder(x)

        # Take the last sequence element
        x = x[-1, :, :]

        # Fully connected layer
        x = self.fc(x)
        return x

# Step 2: Define a custom dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = torch.tensor(self.data.iloc[idx:idx + self.sequence_length, :-1].values, dtype=torch.float)
        y = torch.tensor(self.data.iloc[idx + self.sequence_length, -1], dtype=torch.float).view(1)
        return x,y


seed = 42
torch.manual_seed(seed)

# Step 4: Split the data into training and testing sets
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]
# Extracting features and target from train and test sets
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

**Feature Selection using XGBoost**
seed = 42
torch.manual_seed(seed)

def feature_selector(X_train, y_train,min_features=4,max_features=None):
    if max_features is None:
      max_features = X_train.shape[1]
    best_features = None
    best_mape = np.inf

    # Generate all possible combinations of features
    for n_features_to_select in tqdm(range(min_features, max_features + 1)):
        feature_combinations = combinations(X_train.columns, n_features_to_select)
        for feature_combo in feature_combinations:
            features_subset = X_train[list(feature_combo)]

            # Train GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
            model.fit(features_subset, y_train)

            # Make predictions on the training set
            y_pred = model.predict(features_subset)

            # Calculate MAPE
            mape = mean_absolute_percentage_error(y_train, y_pred)

            # Update best MAPE and best features
            if mape < best_mape:
                best_mape = mape
                best_features = list(feature_combo)
                print('Best MAPE : {0} \n Best Features : {1}'.format(best_mape,best_features))

    return best_features, best_mape
seed = 42
torch.manual_seed(seed)

best_features,best_score = feature_selector(X_train,y_train)
 0%|          | 0/5 [00:00<?, ?it/s]Best MAPE : 2.6513187012522117 
 Best Features : ['HR', 'MO', 'QV2M', 'RH2M']
Best MAPE : 2.6054316598627523 
 Best Features : ['HR', 'MO', 'QV2M', 'PRECTOTCORR']
Best MAPE : 2.3374304949925464 
 Best Features : ['HR', 'MO', 'QV2M', 'T2M']
Best MAPE : 2.0860452719122047 
 Best Features : ['HR', 'MO', 'QV2M', 'WD10M']
 20%|██        | 1/5 [02:33<10:14, 153.73s/it]Best MAPE : 2.024002180678755 
 Best Features : ['HR', 'MO', 'QV2M', 'PRECTOTCORR', 'WD10M']
 40%|████      | 2/5 [04:42<06:57, 139.15s/it]Best MAPE : 2.006800504400937 
 Best Features : ['HR', 'MO', 'QV2M', 'PRECTOTCORR', 'WD10M', 'WS10M']
100%|██████████| 5/5 [06:28<00:00, 77.66s/it]

print(f'Best Features: {best_features}\nBest MAPE: {best_score}')
Best Features: ['HR', 'MO', 'QV2M', 'PRECTOTCORR', 'WD10M', 'WS10M']
Best MAPE: 2.006800504400937
train_data_f = train_data[best_features+['target']]
test_data_f = test_data[best_features+['target']]
**Initializing Parameters and Creating data loaders**
seed = 42
torch.manual_seed(seed)

# Step 5: Create DataLoader instances for training and testing
sequence_length = 10  # You can adjust this
batch_size = 32
train_dataset = TimeSeriesDataset(train_data_f, sequence_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TimeSeriesDataset(test_data_f, sequence_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 6: Initialize and train the model
input_size = 6
hidden_size = 64
num_layers = 2
num_classes = 1
# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNBiLSTMTransformer(input_size, hidden_size, num_layers, num_classes)
model = model.to(device)

best_val_loss = float('inf')
best_model_state = None

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Early stopping parameters
patience = 10
best_test_loss = np.Inf
counter = 0

lr_scheduler_step_size = 5  # Reduce LR after every 5 epochs
lr_scheduler_gamma = 0.1  # Factor by which to reduce LR

scheduler = StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)

seed = 42
torch.manual_seed(seed)

# Training loop
num_epochs = 100
train_losses = []

test_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        inputs = inputs.permute(0, 2, 1)
        inputs = inputs.float()
        outputs = model(inputs)
        targets = targets.float()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            inputs = inputs.permute(0, 2, 1)
            inputs = inputs.float()
            outputs = model(inputs)
            targets = targets.float()
            loss = criterion(outputs, targets)
            test_loss += loss.item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

    scheduler.step()  # Update the learning rate scheduler
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Learning Rate: {scheduler.get_last_lr()}')

    # Check for early stopping
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model_state = model.state_dict()
        torch.save(best_model_state, 'best_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Step 7: Plot training and testing errors
pd.DataFrame(train_losses).to_csv('Bwp Data Train_loss.csv')
pd.DataFrame(test_losses).to_csv('Bwp Data Test_loss.csv')

plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Testing Loss')
plt.legend()
plt.title('Training and Testing Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlim(left=0,right=20)
plt.ylim(top=0.20)
plt.savefig('Loss_curve.png')
plt.show()


seed = 42
torch.manual_seed(seed)

output_list = []
target_list = []
model.eval()
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)  # Move inputs to GPU
        targets = targets.to(device)  # Move targets to GPU

        inputs = inputs.permute(0, 2, 1).float()
        outputs = model(inputs)
        targets = targets.float()
        output_list.extend(outputs.cpu().numpy())  # Move outputs to CPU and append to list
        target_list.extend(targets.cpu().numpy())  # Move targets to CPU and append to list

# Concatenate the lists to get the full outputs and targets arrays
full_outputs = np.concatenate(output_list)
full_targets = np.concatenate(target_list)

# Calculate MAPE
mape = mean_absolute_percentage_error(full_targets, full_outputs)
print(f'MAPE on test set: {mape:.4f}')
    
**Inverse Transsformation**

def invTransform(scaler, data, colName, colNames):
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values

output_list = invTransform(scaler,np.array(output_list,dtype=object).squeeze(),colName='ALLSKY_SFC_SW_DWN',colNames=continuous_features.columns)
target_list = invTransform(scaler,np.array(target_list,dtype=object).squeeze(),colName='ALLSKY_SFC_SW_DWN',colNames=continuous_features.columns)

# Step 8: Create a plot of actual vs. predicted values for the test set
pd.DataFrame(np.array(target_list,dtype=object).squeeze()).to_csv('BWP Data CNN-BLSTM-T Actual values.csv')
pd.DataFrame(np.array(output_list,dtype=object).squeeze()).to_csv('BWP Data CNN-BLSTM-T Predicted values.csv')

plt.figure(figsize=(12, 6))
plt.plot(np.array(target_list,dtype=object).squeeze(), label='Actual', marker='o')
plt.plot(np.array(output_list,dtype=object).squeeze(), label='Predicted', marker='x')
plt.legend()
plt.title(f'Test Set Actual vs. Predicted (MAPE: {mape:.2f}%)')
plt.xlabel('Time Steps')
plt.ylabel('Output Variable')
plt.ylim(bottom=-1,top=2.5)
plt.savefig('Test_Vs_Predicted.png')
plt.show()

from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error,r2_score

# Calculate MAPE on test set
mape = mean_absolute_percentage_error(full_targets, full_outputs)
mae = mean_absolute_error(full_targets, full_outputs)
rmse = mean_squared_error(full_targets, full_outputs,squared=False)
r2_error = r2_score(full_targets, full_outputs)
mse = mean_squared_error(full_targets,full_outputs)

print('MAPE: ',mape)
print('MAE: ',mae)
print('MSE: ',mse)
print('R2 Score: ',r2_error)
print('RMSE Score: ',rmse)

MAPE:  0.89804435
MAE:  0.12999213
MSE:  0.050747152
R2 Score:  0.946514630038048
RMSE Score:  0.22527128

**Architecture Visualization and Summary**
batch = next(iter(train_loader))[0].permute(0, 2, 1).to(device)
yhat = model(batch) # Give dummy batch to forward().
print(model)
CNNBiLSTMTransformer(
  (cnn): Sequential(
    (0): Conv1d(6, 32, kernel_size=(3,), stride=(1,))
    (1): ReLU()
    (2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (bilstm): LSTM(32, 64, num_layers=2, batch_first=True, bidirectional=True)
  (transformer): TransformerEncoderLayer(
    (self_attn): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
    )
    (linear1): Linear(in_features=128, out_features=2048, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (linear2): Linear(in_features=2048, out_features=128, bias=True)
    (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (dropout1): Dropout(p=0.1, inplace=False)
    (dropout2): Dropout(p=0.1, inplace=False)
  )
  (transformer_encoder): TransformerEncoder(
    (layers): ModuleList(
      (0-1): 2 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
        )
        (linear1): Linear(in_features=128, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=128, bias=True)
        (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (fc): Linear(in_features=128, out_features=1, bias=True)
)
