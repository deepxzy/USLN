from PIL import Image
import os
import numpy as np
import torch

from model import USLN
from SegDataset import read_file_list
from tqdm import trange





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = USLN()

model.load_state_dict(torch.load(r'logs/UFO.pth'))

model.eval()
model = model.to(device)

test, path_list_images_test= read_file_list( type='test')



for id in trange(len(test)):
    image = Image.open(test[id]).convert('RGB')
    input = np.transpose(np.array(image, np.float64),(2,0,1))
    input=input/255
    input = torch.from_numpy(input).type(torch.FloatTensor)
    input = input.to(device)
    input= input.unsqueeze(0)
    output = model(input)
    output_np=output.cpu().detach().numpy().copy()
    output_np=output_np.squeeze()
    predictimag=np.transpose(output_np, [1, 2, 0])*255
    a=Image.fromarray(predictimag.astype('uint8'))
    a.save(os.path.join(r"datasets/pred", path_list_images_test[id]))


