#!/usr/bin/env python
# -*- coding: utf-8 -*-


#prim�㷨
def update(index,nodepair,G,n):
#����ʵ�ִ��뿪ʼ
    #��������д����Ĵ���  
    # ���½ڵ�����Ԫ����
    for i in range(n):
        if G[index][i] != 0 and visit[i] == -1:
            if G[index][i] < nodepair[i][1]:
                nodepair[i] = [index, G[index][i]]
#����ʵ�ִ������                                
    return nodepair

def prim(G, n):
    INF      = 100000
    visit    = [-1] * n  #���������¼����
    nodepair =[[-1,INF] for i in range(n)]  #�ڵ�����Ԫ���� ��һ����ʾǰһ������preIndex���ڶ�����ʾ����
    #��ȡ��һ���ڵ�
    visit[0]=1
    print(0,nodepair[0]) #��ӡ��һ���ڵ�
    #���½ڵ�����Ԫ����
    nodepair=update(0,nodepair,G,n)
    #print(nodepair)
    #����ʵ�ִ��뿪ʼ
    #��������д����Ĵ���  
    # ��ѭ����ֱ�����нڵ㶼������
    for _ in range(1, n):
        min_dist = INF
        min_index = -1
        
        # �ҵ���ǰδ���ʽڵ��о�����С�Ľڵ�
        for i in range(n):
            if visit[i] == -1 and nodepair[i][1] < min_dist:
                min_dist = nodepair[i][1]
                min_index = i
        
        # ��Ǹýڵ�Ϊ�ѷ���
        visit[min_index] = 1
        print(min_index, nodepair[min_index])
        
        # ���½ڵ�����Ԫ����
        nodepair = update(min_index, nodepair, G, n)
      
    #����ʵ�ִ������ 
                
    return nodepair
 


if __name__=='__main__':
        Glist=input().split('-')
        G=[]
        for item in Glist:
                row=item.split(',')
                g=[int(i) for i in row]
                G.append(g)
        n=len(G)
        visit = [-1] * n  #���������¼����
        prim(G,n)