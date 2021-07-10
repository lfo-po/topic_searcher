import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

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
        print("Words including the string: '"+string+"':")
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
        print('Printing maximum '+str(max_w)+' words per topic.\n')
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