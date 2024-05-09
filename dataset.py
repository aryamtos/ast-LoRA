import os
from torch.utils.data import Dataset
import numpy as np
import soundfile
from transformers import AutoFeatureExtractor, AutoProcessor
import  librosa
import torchaudio, torch


class Spotify(Dataset):

    def __init__(self, data_path, max_len_AST, split, apply_SpecAug=False, few_shot=False,samples_per_class=1):

        self.max_len_AST = max_len_AST
        self.split = split
        self.data_path = data_path
        self.apply_SpecAug = apply_SpecAug

        self.freq_mask = 24
        self.time_mask = 80

        self.x, self.y = self.get_data()

        if few_shot:
            self.x, self.y = self.get_few_shot_data(samples_per_class)

    def __len__(self):
        return len(self.y)


    def __getitem__(self,index):

        if self.apply_SpecAug:

            freqm = torchaudio.transforms.FrequencyMasking(self.freq_mask)
            timem = torchaudio.transforms.TimeMasking(self.time_mask)


            fbank = torch.transpose(self.x[index],0,1)
            fbank = fbank.unsqueeze(0)

            fbank = freqm(fbank)
            fbank = timem(fbank)

            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank,0,1)

            return fbank, self.y[index]

        else:
            return self.x[index],self.y[index]

    def get_few_shot_data(self,samples_per_class:int):

        x_few,y_few = [],[]

        total_classes = np.unique(self.y)

        for class_ in total_classes:

            cap = 0

            for index in range(len(self.y)):
                if self.y[index] == class_:
                    x_few.append(self.x[index])
                    y_few.append(self.y[index])
                    
                    cap += 1
                    if cap == samples_per_class: break
        return x_few, y_few

    
    def get_data(self):


        processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593",max_length=self.max_len_AST)

        x,y = [],[]

        with open(self.data_path) as f:

            lines = f.readlines()[1:]

        for line in lines:
   
            items = line[:-1].split(',')
            pathh = items[1]
            wav,sampling_rate = soundfile.read(pathh)
            # wav_np = wav.numpy()
            label = items[2]

            x.append(processor(wav,sampling_rate=16000, return_tensors='pt')['input_values'].squeeze(0))
            y.append(self.class_ids[label])
        return np.array(x), np.array(y)

        
    @property
    def class_ids(self):

        return{

            'ba':0,
            're':1,
            'sp':2,
            'mg':3,
            'rj':4
        }



   





