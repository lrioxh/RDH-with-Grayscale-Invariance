import numpy as np
import cv2
from huffman import createTree,walkTree_VLR,encodeImage,decodeHuffmanByDict

def encode_str(msg):
    '''字符串转二进制ascii'''
    return ''.join([f"{ord(i):08b}" for i in msg])


def decode_str(msg):
    return ''.join([chr(int(i, 2)) for i in (msg[i:i + 8] for i in range(0, len(msg), 8))])

def encode_img(img):
    '''二值图像转二进制'''
    x,y=img.shape
    return "{0:0>10b}{1:0>10b}".format(x,y)+''.join(map(str,img.ravel()))
    # return ''.join(map(str,img.ravel()))

def decode_img(bin):
    l=10
    x=int(bin[:l],2)
    y=int(bin[l:l*2],2)
    img=np.array([np.int8(i) for i in bin[l*2:]])
    err=x*y-img.size
    if err>0:
        print('fail to decode')
        # print(f'add {err} 1s')
        # img=np.concatenate((img,[1]*err))

    return img.reshape(x,y)


def ncc(img1,img2):
    '''计算图像归一化相关'''
    return np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2))


tree4c={'0000':'0','1111':'100',
        '0001':'1010','1000':'1011','0011':'1100','1100':'1101','0111':'1110','1110':'11110',
        '0110':'1111100',
        '1001':'11111010','0010':'11111011','0100':'11111100','1011':'11111101','1101':'11111110',
        '0101':'111111110','1010':'111111111'
        }


def huffman_encode4(stream):
    '''预定义的霍夫曼树编码,适用于连续01,且0较多'''
    # tree4=dict.fromkeys(tree4c.keys, 0)
    stream=''.join(['0' if m=='1' else '1' for m in stream])
    lenth=len(stream)
    res=''
    r=lenth%4
    res+=f"{r:02b}"     # 2位记录余数，单独处理
    if r!=0:
        res+=stream[:r]
        stream=stream[r:]
        lenth=len(stream)
    for i in range(0,lenth,4):
        ii=stream[i:i+4]
        res+=tree4c[ii]
        # tree4[ii]+=1
    return res

def huffman_decode4(stream):
    ivtree = {v: k for k, v in tree4c.items()}
    res=''
    # print(stream[:2])
    r=int(stream[:2],2)
    if r!=0:
        res+=stream[2:2+r]
        stream=stream[2+r:]
    else:
        stream=stream[2:]
    lenth=len(stream)
    i=0
    hight=[1,3,4,5,7,8,9]
    while i < lenth:
        for h in hight:
            if stream[i:i+h] in ivtree:
                res+=ivtree[stream[i:i+h]]
                i+=h
                break
    
    res=''.join(['0' if m=='1' else '1' for m in res])
    return res


def golomb_encode2(stream,slide0=8,slide1=4):
    '''一种游程编码,2位标志位00表示slide0位全0,01表示slide0位记录0的游程,
    10表示slide1位记录1的游程,11表示1'''
    # lenth=len(stream)
    res=''
    count0=0
    count1=0
    # while i<len(stream):
    #     i+=1
    for s in stream:
        if s=='0':
            count0+=1
            if count1>1:
                res+='10'
                res+=f"{count1:0{slide1}b}"
                count1=0
            elif count1==1:
                res+='11'
                count1=0

            if count0==2**slide0:
                res+='00'
                count0=0
        else:
            count1+=1
            if count0>0:
                res+='01'
                res+=f"{count0:0{slide0}b}"
                count0=0

            if count1==2**slide1-1:
                res+='10'
                res+='1'*slide1
                count1=0
    if  count0>0:
        res+='01'
        res+=f"{count0:0{slide0}b}"
        count0=0
    elif count1>1:
        res+='10'
        res+=f"{count1:0{slide1}b}"
        count1=0
    elif count1==1:
        res+='11'
        count1=0
    return res

def golomb_decode2(stream,slide0=8,slide1=4):
    res=''
    i=0
    while i<len(stream)-1:
        type=stream[i:i+2]
        i+=2
        if type=='00':
            res+='0'*2**slide0
        elif type=='01':
            count=stream[i:i+slide0]
            if count:
                res+='0'*int(count,2)
            else: print('error out of index')
            i+=slide0
        elif type=='10':
            count=stream[i:i+slide1]
            if count:
                res+='1'*int(count,2)
            else: print('error out of index')
            i+=slide1
        else:
            res+='1'
    return res

def compression_encode(stream,fix=4):
    '''一种游程编码'''
    res=''
    count0=0
    count1=0
    i=0
    while i<len(stream):
    # iter=enumerate(stream)
    # for i,s in iter:
        if stream[i]=='0':
            count0+=1
            if count1==0:i+=1
            elif count1<4:
                res+='0'
                res+=stream[i-count1:i-count1+fix]
                i=i-count1+fix
                count1=0
                count0=0
                # next(iter)
            else:
                l=int(np.log2(count1))
                ll=(l-1)*'1'+'0'
                res+=ll
                res+=f"{count1-2**l:0{l}b}1"
                count1=0
                i+=1
            
        else:
            count1+=1
            if count0==0:i+=1
            elif count0<4:
                res+='0'
                res+=stream[i-count0:i-count0+fix]
                i=i-count0+fix
                count0=0
                count1=0
            else:
                l=int(np.log2(count0))
                ll=(l-1)*'1'+'0'
                res+=ll
                res+=f"{count0-2**l:0{l}b}0"
                count0=0
                i+=1
    if count1>=4:
        l=int(np.log2(count1))
        ll=(l-1)*'1'+'0'
        res+=ll
        res+=f"{count1-2**l:0{l}b}1"
    elif count1>0:
        res+='0'
        res+=stream[i-count1:]
    elif count0>=4:
        l=int(np.log2(count0))
        ll=(l-1)*'1'+'0'
        res+=ll
        res+=f"{count0-2**l:0{l}b}0"
    elif count0>0:
        res+='0'
        res+=stream[i-count0:]
    return res
    
def compression_decode(stream,fix=4):
    res=''
    i=0
    while i<len(stream):
        if stream[i]=='0':
            i+=1
            res+=stream[i:i+fix]
            i+=fix
        else:
            j=i
            while stream[i]=='1':i+=1
            l=i-j+1
            i+=1
            count=2**(l)+int(stream[i:i+l],2)
            i+=l
            res+=count*stream[i]
            i+=1
    return res

def huffman_encode(msg):
    '''使用根据比特流构建的霍夫曼树编码'''
    i=0
    imgcode=[]
    while i<len(msg):
        # imgcode.append(''.join([str(j) for j in msg[i:i+4]]))
        imgcode.append(msg[i:i+4])
        i+=4
    imgcode=np.array(imgcode)
    hist_dict = {}
    global Huffman_encode_dict
 
    # 得到原始图像的直方图，出现次数为0的元素(像素值)没有加入
    for p in imgcode:
        if p not in hist_dict:
            hist_dict[p] = 1
        else:
            hist_dict[p] += 1
    huffman_root_node = createTree(hist_dict)
    Huffman_encode_dict=walkTree_VLR(huffman_root_node)
    res = encodeImage(imgcode, Huffman_encode_dict)
    return res

def huffman_decode(msg):
    global Huffman_encode_dict
    mg_src_val_array = decodeHuffmanByDict(msg, Huffman_encode_dict)
    res=''.join(mg_src_val_array)
    return res

if __name__ == '__main__':
    # s="00101100111111111000000000111111"
    # golomb_encode2
    # huffman_encode4
    # huffman_encode
    # compression_encode
    # s_=huffman_encode4(s)
    # print(len(s_)/len(s),s==huffman_decode4(s_))

    msg=cv2.imread('./data/msg.jpg',cv2.CV_8UC1)
    # cv2.imshow('1',msg)
    # cv2.waitKey(0)
    # print(msg.shape)
    ret, msg = cv2.threshold(msg, 200, 255, cv2.THRESH_BINARY)
    msg[msg>0]=1
    msg=encode_img(msg)
    msgC=huffman_encode4(msg)
    print(len(msgC)/len(msg),msg==huffman_decode4(msgC))
    msgC=compression_encode(msg)
    print(len(msgC)/len(msg),msg==compression_decode(msgC))
    