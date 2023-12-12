import os
import numpy as np
from .glove_simi import generate_glove, AD_DIRECTORY

class AD_Distribution():
    def __init__(self, simi_name, alpha, beta):
        print('using ........ ', simi_name, ' .... knowledge')
        path_simi_name = os.path.join(AD_DIRECTORY, 'imagenet_cos_similarity')
        file_simi_name = path_simi_name +'_'+ simi_name + '.npy'
        if os.path.exists(file_simi_name):
            self.cos_similarity = np.load(file_simi_name)
            print(simi_name+" cos_similarity loaded")
        else:
            self.cos_similarity = self.generate_similarity(simi_name)
        
        self.alpha = alpha
        self.beta  = beta
    
    def generate_similarity(self,simi_name):
        if simi_name == 'glove':
            similarity = generate_glove()
        else:
            print(simi_name + 'not implemented yet')
        return similarity

    def generate_distribution(self, gt_label, target):
        distribution=[]

        for i in range(len(target)):
            distri = self.single_distribution_build(i, target[i], gt_label[i])
            distribution.append(distri)

        distribution = np.array(distribution)
        return distribution
    
    def single_distribution_build(self,index, target_id, gt_id):
        if target_id.shape == ():
            target_id = np.array([target_id])

        simil_logits = np.zeros(self.cos_similarity[target_id[0],:].shape)
        for i in range(len(target_id)):
            simil_logits += self.cos_similarity[target_id[i],:]
            
        simil_logits = (simil_logits)/ len(target_id)
        logit_value = self.alpha

        for i in range(len(target_id)):
            simil_logits[target_id[i]] = logit_value
            logit_value = logit_value - self.beta 
            
        if not self.check_oreder_target_no_groundtruth(simil_logits, target_id):
            print('fail to generate distribution for index: ', index) 
            
        logits = self.softmax(simil_logits)
        return logits

    def check_oreder_target_no_groundtruth(self, probs, target_id):
        sort_labels = np.argsort(probs)
        cnt = 0
        for i in range(len(target_id)):
            if target_id[-(i+1)] == sort_labels[-(len(target_id)-i)]: 
                cnt +=1
        if (cnt == len(target_id)):
            return True
        else:
            return False

    def softmax(self,logits):
        
        prob=np.exp(logits) / np.sum(np.exp(logits))
        return prob

