import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import seaborn as sns
import matplotlib.pyplot as plt

def get_distributions_data(corpus,element,dist_kind,hl,path_data):
    '''
    In:
        - corpus (string), label for corpus of documents
        - element (string), words: 'w', articles: 'a', keywords: 'kw'
        - dist_kind (string), either 'docs-topics' or 'topics-elements'
        - hl (int): hierarchical level for the topic model sbm
        Optional:
           - path_data (string): to specify the path where to find the distributions files.
    Out: 
        - dist: (2-D numpy array) either a distribution over elements for each topic (when dist_kind=topics-elements), or a
          distribution over topics for each decision (when dist_kind=topics-elements).
    
    Gets distribution data for a specified corpus and element.
    '''
    
    if dist_kind=='topics-elements':
        dist_name='p-t'+element+'-'+element
    elif dist_kind=='docs-topics':
        dist_name='p-d-t'+element
    else:
        dist_name=None
        print('Error in specifying distribution: Should be either topics-elements or docs-topics.')
              
    filename=corpus+'-d'+element+'_'+dist_name+'_l-'+str(hl)+'.npz'

    dist_csr=load_npz(path_data+filename)
    dist=dist_csr.toarray()    

    return dist

def get_topic_common_elements(t,p_te_e,element_names,Ne,get_values=False):
    top_idx=(-p_te_e[t]).argsort()[:Ne]
    
    if get_values:
        top_elem={}
    else:
        top_elem=[]
    for i in range(Ne):
        if p_te_e[t][top_idx[i]]>0.:
            if get_values:
                top_elem[element_names[top_idx[i]]]=p_te_e[t][top_idx[i]]
            else:
                top_elem.append(element_names[top_idx[i]])
                
    return top_elem 

def find_elements(string, element_list, min_ch=4):
    e_found=[]
    if type(string)==str and len(string)>=min_ch:
        print("Words/Articles including the string: '"+string+"':")
        pt=re.compile(string,flags=re.UNICODE)
        for e in element_list:
            if pt.search(e) is not None:
                print(e)
                e_found.append(e)
        if len(e_found)==0:
            print('   Zero results found.')
    else:
        print('String not valid. Should be at least 4 character length.')
        
def get_element_topic_info(e,topic_memb,memberships_df,element_list,element,corpus,max_w=15,path='./topic_model_data/'):
    if e in element_list:    
        if element=='word':
            print('Printing maximum '+str(max_w)+' words per topic.\n')
        else:
            print('Printing maximum '+str(max_w)+' articles per topic.\n')
        for i in range(1,len(topic_memb)):
            p_te_e=get_distributions_data(corpus,element,'topics-elements',i-1,path) 
            Nt=len(p_te_e)
            t=memberships_df.loc[e]['membership_l'+str(i)]
            print('Hierarchical level: '+str(i)+'. Topic '+str(t)+'/'+str(Nt)+'.')

            t_details=topic_memb[i][t]
            t_down=t_details['topics_l'+str(i-1)]
            for td in t_down:
                print('    Level-'+str(i-1)+' topic: '+str(td))
                w=get_topic_common_elements(td,p_te_e,element_list,15,get_values=False)
                for w_i in w:
                    if w_i==e:
                        print('    \033[1m' + w_i + '\033[0m')
                    else:    
                        print('    '+w_i)    
                print('    ---------------\n')
            print('\n\n')
    else:
        print('The element '+e+' was not found, try again.')  
        
def topic_years_distribution(Y,corpus,element,l,df,path_data,add_pseudo_counts=False):
    '''
    In:
        - Y (a list of integers) natural years list (contiguous)
        - corpus (string), label for corpus of documents
        - element (string), words: 'w', articles: 'a', keywords: 'kw'
        - df (pandas DataFrame) should contain the columns: 'date_years', 'idx' and the decision ids as indices of the DF
        - add_pseudo_counts (optional)  
    Out: 
        - A 2D numpy array of dimension number of topics x number of years. For each row (topic) the distribution over 
        the years. 
        
    The function computes the distribution of topics in each year and then transposes the result. Hence, the normalizations 
    are over topics and not over years.
    
    When add_pseudo_counts=true, the distribution over topics for each specific year is modified according the add_pseudocunts 
    function
    ''' 
    
    p_d_te=get_distributions_data(corpus,element,'docs-topics',l,path_data)  
    
    p_y_t=[]
    for y in Y:
        dist,N=topic_distribution_time_interval((y,y+1),p_d_te,df)
        if add_pseudo_counts:
            dist=add_pseudocounts(dist,Nw*N)
            p_y_t.append(dist)
        else:
            p_y_t.append(dist)

    p_t_y=np.transpose(np.array(p_y_t))     
    
    return p_t_y 

def topic_distribution_time_interval(y,p_d_te,df):
    '''
    In:
        - y (tuple, 2 values), time limits in years.
        - p_d_te (numpy array of dimension 2: number of decisions x number of topic-elements), at each row (for each decision)
          the distribution over topics.
        - df (pandas DataFrame) should contain the columns: 'date_years', 'idx' and the decision ids as indices of the DF
    Out: 
        - d_mean: mean distribution of topics a long decisions in the time interval given. numpy array of size the number 
          of topics. 
        - An integer: the number of decisions used to compute the mean
    
    The function computes the mean over all decisions topic distributions in a time interval
    '''
        
    decisions=df.loc[(df.date_years>y[0]) & (df.date_years<=y[1])].index
#     print(decisions)
    
    dists=[]
    nan_count=0
    for d in decisions:
        idx=df.loc[d].idx
        #print(idx)
        if np.isnan(p_d_te[idx][0]):
            nan_count+=1
        else:
            dists.append(p_d_te[idx])
#         print(d)
#         print(p_d_tw[idx])    
#         print(' ')
    d_mean=np.mean(dists,axis=0)    
    return d_mean,len(decisions)-nan_count


def get_topic_common_decisions(t,hl,d_df,element,corpus,path='./topic_model_data/',Nd=10):
    p_d_te=get_distributions_data(corpus,element,'docs-topics',hl,path)
    if t>len(p_d_te[0]) or t<0:
        print('There are '+str(len(p_t_y))+' topics at level '+str(hl)+'. Topic should be within this range.')
        top_d_idx=[]
    else:    
        p_te_d=np.transpose(p_d_te)
        top_d_idx=(-p_te_d[t]).argsort()[:Nd]
        
        print('Decisions where topic '+str(t)+' (level '+str(hl)+') appears the most:')
        for d in list(d_df.iloc[top_d_idx].index):
            print('TOL_'+str(d))
            
            
def plot_topic_evolution(hl,t,corpus,element,topic_memb,path):
    y1=2001
    y2=2018
    years=np.arange(y1,y2+1)
    
    p_t_y=np.load(path+corpus+'-'+element+'_years_distribution_hl-'+str(hl)+'.npy')
    p_t_y_sub=np.load(path+corpus+'-'+element+'_years_distribution_hl-'+str(hl-1)+'.npy')
    
    if t<len(p_t_y):
        sub_topics=topic_memb[hl][t]['topics_l'+str(hl-1)]
        st_colors=sns.color_palette('deep',n_colors=len(sub_topics))
        t_evo=p_t_y[t]
        m=np.max(t_evo)
        t_evo=t_evo#/m
        pos=np.argmax(t_evo)

        st_evo=[]
        st_pos=[]
        for st in sub_topics:
            evo=p_t_y_sub[st]
            m=np.max(evo)
            st_evo.append(evo)#/m)
            st_pos.append(np.argmax(evo))


        fig, ax  = plt.subplots(1, 2,
                            figsize=(16, 7),
                            gridspec_kw={#'wspace':0.4,
                                         #'hspace':0.1,
                                         'width_ratios': [5, 5]}
                           )

        ax[0].plot(years,t_evo,'-',lw=2,color='black')
        ax[0].text(years[pos],np.max(t_evo),'Topic '+str(t))
        for i in range(len(sub_topics)):
            ax[1].plot(years,st_evo[i],'-',color=st_colors[i],alpha=0.6)
            ax[1].text(years[st_pos[i]],np.max(st_evo[i]),'Topic '+str(sub_topics[i]),color=st_colors[i])    

        for i in range(2):
            ax[i].set_ylabel('Topic importance') #(normalized to 1)
            ax[i].set_xlabel('Years')
            ax[i].set_xticks(years)
            ax[i].set_xticklabels(years,rotation=45)
        ax[0].set_title('Topic importance evolution')
        ax[1].set_title('Sub-Topics (level '+str(hl)+') importance evolution')
        plt.show() 
    else:
        print('There are '+str(len(p_t_y))+' topics at level '+str(hl)+'. Topic should be within this range.')