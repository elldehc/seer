import pandas as pd
import torch
import numpy as np
import argparse
import time
import random

def parse_args():
    parser = argparse.ArgumentParser(description="seer")
    parser.add_argument('--lr', type=float, default=0.001,help='lr')
    parser.add_argument('--n_epoch', type=int, default=40, help='n_epoch')
    parser.add_argument('--ker_size', nargs='?', default='[20]',help='ker_size')
    parser.add_argument('--pool_size', type=int, default=2, help='pool_size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--conv_out_size', nargs='?', default='[20]',help='conv_out_size')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden_size')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    return parser.parse_args()

predict_len=20
predict_adv=1
data_limit=1
name_len=0
kpi_len=0
args=parse_args()
n_train=0
n_test=0
max_rate=0
max_f=0


def getdata():
    global name_len,kpi_len,n_train,n_test,train_in,train_ans,test_in,test_ans
    with open("../data/processed/namelist.txt","r")as fin:
        names=eval(fin.read())
    l=[]
    for it in names:
        t=np.array(pd.read_csv("../data/processed/train_"+it+".csv",index_col="timestamp").fillna(0))[:-predict_adv*data_limit:data_limit]
        l.append(t)
        print(it)
        #print(t.shape)
    l=np.array(l,dtype=np.float32)
    name_len=l.shape[0]
    kpi_len=l.shape[2]
    train_in=np.transpose(l,[1,2,0])
    #print(train_in.dtype)
    print("train_in ok")
    train_ans=np.array(pd.read_csv("../data/processed/train_ans.csv",index_col="timestamp").fillna(0))[(predict_len+predict_adv)*data_limit::data_limit]
    for i in range(train_ans.shape[0]):
        if train_ans[i][0]>name_len:
            train_ans[i][0]=0
    train_ans=torch.as_tensor(train_ans,dtype=torch.int64).reshape([-1]).cuda()
    print("train_ans ok")
    print(train_in.shape[0],train_ans.shape[0])
    n_train=train_ans.shape[0]
    l=[]
    for it in names:
        t=np.array(pd.read_csv("../data/processed/test_"+it+".csv",index_col="timestamp").fillna(0))[:-predict_adv*data_limit:data_limit]
        l.append(t)
        print(it)
    l=np.array(l,dtype=np.float32)
    test_in=np.transpose(l,[1,2,0])
    #print(test_in.dtype)
    print("test_in ok")
    test_ans=np.array(pd.read_csv("../data/processed/test_ans.csv",index_col="timestamp").fillna(0))[(predict_len+predict_adv)*data_limit::data_limit]
    for i in range(test_ans.shape[0]):
        if test_ans[i][0]>name_len:
            test_ans[i][0]=0
    test_ans=torch.as_tensor(test_ans,dtype=torch.int64).reshape([-1]).cuda()
    print("test_ans ok")
    #print(test_ans.shape)
    n_test=test_ans.shape[0]
    #print(n_train,n_test,name_len,kpi_len)
    ttt=time.time()-tttt
    tmin=int(ttt/60)
    tsec=ttt-tmin*60
    print("elapsed time=%d:%f"%(tmin,tsec))
    
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_out_size=[kpi_len]
        self.conv_out_size.extend(list(eval(args.conv_out_size)))
        self.ker_size=[0]
        self.ker_size.extend(list(eval(args.ker_size)))
        #print(self.conv_out_size)
        #print(self.ker_size)
        self.pool_size=args.pool_size
        self.n_layers=len(self.conv_out_size)
        self.conv=torch.nn.ModuleList([None])
        self.sz=name_len
        for i in range(1,self.n_layers):
            self.conv.append(torch.nn.Conv1d(self.conv_out_size[i-1], self.conv_out_size[i], self.ker_size[i]))
            self.sz=(self.sz-self.ker_size[i]+1)#//self.pool_size
        #print("self.sz=",self.sz)
        # self.conv = torch.nn.Conv1d(300, self.conv_out_size, self.ker_size)
        self.pool = torch.nn.MaxPool1d(self.pool_size)
        self.drop = torch.nn.Dropout(args.dropout)
        self.relu=torch.nn.ReLU()
        self.rnn=torch.nn.LSTM(input_size=self.sz*self.conv_out_size[-1],hidden_size=args.hidden_size,num_layers=args.num_layers,batch_first=True,dropout=args.dropout)
        self.dense = torch.nn.Linear(args.hidden_size,name_len+1)
        
    def forward(self,x):
        
        x_drop = torch.reshape(x,(x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))
        #print(x.shape,x_drop.shape)
        for i in range(1,self.n_layers):
            x_conv = self.conv[i](x_drop)
            x_relu=self.relu(x_conv)
            #x_pool = self.pool(x_relu)
            #x_drop = self.drop(x_pool)
            x_drop = self.drop(x_relu)
        x_resize = torch.reshape(x_drop, (-1,predict_len,self.sz*self.conv_out_size[-1]))
        #print(x_resize.shape)
        x_rnn=self.rnn(x_resize)[0][:,-1]
        x_dense = self.dense(x_rnn)
        y_pred=self.drop(x_dense)
        #print(y_pred.shape)
        return y_pred

def test(epoch):
    global max_rate,max_f
    batch_size=64
    hit_count=0
    tp=[[0,0,0,0] for i in range(name_len+1)]
    for i in range(0,n_test,batch_size):
        real_in=[]
        for j in range(batch_size):
            if i+j>=n_test:
                break
            if(test_in[i+j:i+j+predict_len].dtype!=np.float32):
                print(i,j,test_in[i+j:i+j+predict_len].dtype)
            real_in.append(test_in[i+j:i+j+predict_len])
        real_in=torch.as_tensor(np.array(real_in)).cuda()
        yp = model(real_in)
        for j in range(batch_size):
            if i+j>=n_test:
                break
            t_out=[0 for k in range(name_len+1)]
            #print(test_ans[i+j])
            tans=int(test_ans[i+j])
            t_out[tans]=1
            #print(j,torch.argmax(yp[j]),tans)
            if torch.argmax(yp[j])==tans:
                hit_count+=1
                for k in range(name_len+1):
                    if k==tans:
                        tp[k][0]+=1
                    else:
                        tp[k][3]+=1
            else:
                for k in range(name_len+1):
                    if k==tans:
                        tp[k][2]+=1
                    else:
                        tp[k][1]+=1
    f=0
    prec=0
    rec=0
    for i in range(name_len+1):
        #print(i,tp[i])
        if(tp[i][0]+tp[i][1]==0):
            prec+=1
        else:
            prec+=tp[i][0]/(tp[i][0]+tp[i][1])
        if(tp[i][0]+tp[i][2]==0):
            rec+=1
        else:
            rec+=tp[i][0]/(tp[i][0]+tp[i][2])
    if prec!=0 or rec!=0:
        f=2.0*prec*rec/(prec+rec)/(name_len+1)
    ttt=time.time()-tttt
    tmin=int(ttt/60)
    tsec=ttt-tmin*60
    if(epoch>=0):
        if (args.verbose>0):
            print("epoch=%d hit count=%d ratio=%f f-score=%f" % (epoch,hit_count,hit_count/n_test,f),"elapsed time=%d:%f"%(tmin,tsec))
        if hit_count/n_test>max_rate:
            max_rate=hit_count/n_test
            max_f=f
            torch.save(model, "../saved_models/"+args.ker_size+'_'+args.conv_out_size+'_'+str(args.hidden_size)+'_'+str(args.num_layers)+'_'+time_start+".dat")
    else:
        if (args.verbose>0):
            print("init: hit count=%d ratio=%f f-score=%f" % (hit_count,hit_count/n_test,f),"elapsed time=%d:%f"%(tmin,tsec))
        if hit_count/n_test>max_rate:
            max_rate=hit_count/n_test
            max_f=f
            
if __name__=="__main__":
    tttt=time.time()
    time_start=time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
    lr=args.lr
    batch_size=32
    n_epoch=args.n_epoch
    getdata()
    random.seed()
    model=Model().cuda()
    loss_c=torch.nn.CrossEntropyLoss().cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    test(-1)
    for epoch in range(n_epoch):
        shuffle_list=list(range(train_ans.shape[0]))
        random.shuffle(shuffle_list)
        for i in range(0,n_train,batch_size):
            real_in=[]
            for j in shuffle_list[i:i+batch_size]:
                real_in.append(train_in[j:j+predict_len])
                #print(train_in[j:j+predict_len].dtype)
            real_in=torch.as_tensor(np.array(real_in)).cuda()
            yp=model(real_in)
            loss=loss_c(yp,train_ans[shuffle_list[i:i+batch_size]])
            #if i%(batch_size*10)==0:
                #print("epoch=%d i=%d loss=%f"%(epoch,i,loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        test(epoch)
    print("best ratio=%f f-score=%f"%(max_rate,max_f))
    
    
        