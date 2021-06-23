from train import *

device = torch.device('cuda')

model = Machine(batch_size=1)
model = model.to(device)
model.train()
model.reset()

state = torch.load('model.pth')
model.load_state_dict(state['model'])

data = Dataset(batch_size=1)
batch_input, batch_output = data.batch()
batch_input, batch_output = batch_input.to(device), batch_output.to(device)
print(batch_input, batch_output, batch_input.shape)

outputs = []
for i in range(batch_input.shape[0]):
    input = batch_input[i]
    model_output = model(input)
    print(input.cpu().numpy())
    print(model_output.cpu().detach().numpy())
    print(batch_output[i].cpu().numpy())
    print()
    outputs.append(model_output)
outputs = torch.stack(outputs, dim=0).detach().cpu().numpy()

#print(outputs)