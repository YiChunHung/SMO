import numpy as np
import csv
import math

#from sklearn import svm
#from sklearn.model_selection import GridSearchCV

## Read the csv file
def read_csv(path):
    data = []
    with open(path,newline='') as csvfile:
        data_csv = csv.reader(csvfile,delimiter=',')
        for row in data_csv:
            data.append(row)
    data = np.array(data,dtype=float)
    return data[:,0], data[:,1:]
def debug(string):
    print(string)


class SVM:

    def __init__(self, tol=0.001, c=1.0,kernel='linear',max_iter=1000):
        self.tol = tol
        self.c = c
        self.max_iter = max_iter
        self.kernel = self.linear_kernel


    #Start to fit the feature to the label
    def fit(self, fea, label):
        if(fea.shape[0] != label.shape[0]):
            debug("The shape of feature and label must be same")
        
        #init the coefficient and F
        label[label==0] = (-1)
        self.lamda = np.zeros(fea.shape[0])
        self.b = 0
        self.f = label * (-1)
        self.fea = fea
        self.label = label


        self.smo(fea,label)
        #print(self.lamda)
        #print(self.b)

    # The smo algorithm starts
    def smo(self,fea,label):
        num_changed = 0
        examine_all = 1
        iter_time = 0
        while ((num_changed > 0 or examine_all==1)) and (iter_time<self.max_iter):
            iter_time += 1
            num_changed = 0
            if (examine_all==1):
                for i in range(fea.shape[0]):
                    num_changed = num_changed + self.examine_example(i,fea,label)
            else:
                i0 = np.where((self.lamda<self.c) & (self.lamda > 0))[0]    #choose I0 to be efficient
                for i in i0:
                    num = self.examine_example(i,fea,label)
                    if(num==-1):
                        num_changed = 0
                        break
                    num_changed = num_changed + num
            if (examine_all == 1):
                examine_all = 0
            else:
                if(num_changed == 0):
                    examine_all = 1

        # Compute the bias term
        sup_vec = np.where((self.lamda<self.c) & (self.lamda > 0))[0]
        if (sup_vec.shape[0]>0):
            sup_vec_x = fea[sup_vec[0],:]
            self.b = label[sup_vec[0]] - np.sum((self.lamda*label) * \
                self.kernel(fea,sup_vec_x)[:,0])
        else:
            debug('C you choose may be too small(large)\n')

    def examine_example(self,i2,fea,label):
        f2 = self.f[i2]
        y2 = label[i2]
        lamda2 = self.lamda[i2]
        i0 = (self.lamda<self.c) & (self.lamda>0) #True mean it is I0 set
        low = (i0) | ((label == 1) & (self.lamda == 0)) | ((label == -1) & (self.lamda==self.c)) 
        up = (i0) | ((label == 1) & (self.lamda == self.c)) | ((label == -1) & (self.lamda==0))
        b_up = min(self.f[low]) + self.tol
        b_low = max(self.f[up]) - self.tol

        if (b_up > b_low):
            return -1

        if( ((low[i2]) & (f2 < b_low)) | ((up[i2]) & (f2 > b_up)) ):
            i0 = (self.lamda > 0) & (self.lamda < self.c)
            i0_idx = np.where(i0==True)[0]
            
            if(sum(i0) > 0):
                i1 = np.argmax(np.abs(self.f[i0_idx]-self.f[i2]))
                if self.takestep(i1,i2,fea,label):
                    return 1

            for i in np.random.permutation(np.shape(i0_idx)[0]):
                if self.takestep(i0_idx[i],i2,fea,label):
                    return 1

            for i in np.random.permutation(np.shape(label)[0]):
                if ~i0[i]:
                    if self.takestep(i,i2,fea,label):
                        return 1
        return 0

    def takestep(self,i1,i2,fea,label):
        if i1==i2:
            return 0
        lamda1 = self.lamda[i1]
        lamda2 = self.lamda[i2]
        y1 = label[i1]
        y2 = label[i2]
        f1 = self.f[i1]
        f2 = self.f[i2]
        s = y1*y2
        if(s==1):
            l = max(0,lamda1+lamda2-self.c)
            h = min(self.c,lamda1+lamda2)
        elif(s==-1):
            l = max(0,lamda2-lamda1)
            h = min(self.c,self.c-lamda1+lamda2)
        else:
            debug("Label must be -1 or 1")
        if(l==h):
            return 0
        k11 = self.kernel(fea[i1,:],fea[i1,:])[0,0]
        k12 = self.kernel(fea[i1,:],fea[i2,:])[0,0]
        k22 = self.kernel(fea[i2,:],fea[i2,:])[0,0]
        eta = k11 + k22 - 2*k12
        if(eta > 0):
            lamda2_new = lamda2 + y2*(f1-f2)/eta
            if (lamda2_new < l):
                lamda2_new = l
            elif (lamda2_new > h):
                lamda2_new = h
        elif(eta==0):
            psi3_l = label[i2]*(f1-f2)*l
            psi3_h = label[i2]*(f1-f2)*h
            if(psi3_h>psi3_l):
                lamda2_new = h
            else:
                lamda2_new = l
        else:
            f1_new = y1*f1 - lamda1*k11 - s*lamda2*k12
            f2_new = y2*f2 - s*lamda1*k12 - lamda2*k22
            l1 = lamda1 + s*(lamda2-l)
            h1 = lamda1 + s*(lamda2-h)
            l_obj = l1*f1_new + l*f2_new + (l1**2)*k11/2 + (l**2)*k22/2 + s*l*l1*k12
            h_obj = h1*f1_new + h*f2_new + (h1**2)*k11/2 + (h**2)*k22/2 + s*h*h1*k12
            if (l_obj < h_obj-self.tol):
                lamda2_new = l
            elif(l_obj > h_obj+self.tol):
                lamda2_new = h
            else:
                lamda2_new = lamda2
        if (abs(lamda2_new-lamda2) < self.tol*(lamda2+lamda2_new+self.tol)):
            return 0
        lamda1_new = lamda1 + s*(lamda2-lamda2_new)
        del_lambda2 = lamda2_new - lamda2
        del_lambda1 = -1*y1*y2*del_lambda2
        #debug(self.linear_kernel(fea[:,:],fea[i1,:])[:,0].shape)
        #debug(self.f.shape)
        self.f = self.f + del_lambda1*y1*self.kernel(fea[:,:],fea[i1,:])[:,0] + \
        del_lambda2*y2*self.kernel(fea[:,:],fea[i2,:])[:,0]
        self.lamda[i2] = lamda2_new
        self.lamda[i1] = lamda1_new
        return 1


    def linear_kernel(self,xi,xj):
        return np.array(np.matrix(xi)*np.matrix(xj).T)


    def predict(self,fea_test):
        trn_size = np.shape(self.fea)[0]
        test_size = np.shape(fea_test)[0]
        label_test = []
        for i in range(test_size):
            label = (np.sum((self.lamda*self.label)*self.kernel(self.fea,fea_test[i,:])[:,0]) + self.b)>=0
            
            if (label == True):
                label_test.append(1)
            else:
                label_test.append(-1)
        return np.array(label_test)
    def coeff(self):
        return np.array(np.matrix(self.lamda*self.label)*np.matrix(self.fea))[0,:], self.b


if __name__ == '__main__':
    
    trn_path = './data/messidor_features_training.csv'
    test_path = './data/messidor_features_testing.csv'

    trn_label,trn_fea = read_csv(trn_path)
    test_label,test_fea = read_csv(test_path)
    trn_label[trn_label==0] = -1
    test_label[test_label==0] = -1

    print('Warning: Choosing n=10 may take lots of time.')
    print("Please choose n fold of the cross validation.(1) n = 5 (2) n = 10.\n")
    n = input()
    if n == '1':
        n = 5
    else:
        n = 10

    c_set = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5]
    #print(trn_fea.shape[0])
    subset_size = math.floor(trn_fea.shape[0]/n)
    scores = []
    
    ## Cross validation
    for c in c_set:
        print('Processing :',c)
        score = 0
        for i in range(n):
            if i==0:
                fea_training = trn_fea[(i+1)*subset_size:,:]
                fea_testing = trn_fea[:(i+1)*subset_size,:]
                label_training = trn_label[(i+1)*subset_size:]
                label_testing = trn_label[:(i+1)*subset_size]
            else:
                training1 = trn_fea[:i*subset_size,:]
                training2 = trn_fea[(i+1)*subset_size:,:]
                fea_training = np.concatenate((training1,training2),axis=0)
                fea_test = trn_fea[i*subset_size:(i+1)*subset_size,:]

                label1 = trn_label[:i*subset_size]
                label2 = trn_label[(i+1)*subset_size:]
                label_training = np.concatenate((label1,label2),axis=0)
                label_testing = trn_label[i*subset_size:(i+1)*subset_size]
            clf = SVM(c=c,max_iter=500)
            clf.fit(fea_training,label_training)
            pred_label = clf.predict(fea_testing)
            score += np.sum(pred_label==label_testing)
            print("accuracy:",np.sum(pred_label==label_testing)/pred_label.shape[0])
        scores.append(score)
        print(score)
    max_score = max(scores)
    max_idx = scores.index(max_score)
    print('The optimal c :',c_set[max_idx])

    print('Use the optimal c to train the entire model.')
    clf = SVM(c=c_set[max_idx],max_iter=1000)
    clf.fit(trn_fea,trn_label)
    pred_label = clf.predict(test_fea)
    w, b = clf.coeff()
    print("H : ", w)
    print('b : ', b)
    print('Accuracy of the model:',np.sum(pred_label==test_label)/pred_label.shape[0]*100,'%')
    





    



    










