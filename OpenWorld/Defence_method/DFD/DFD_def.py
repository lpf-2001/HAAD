from tqdm import tqdm
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
def round_up(n, digits=0):
    return Decimal(str(n)).quantize(Decimal('1e-{0}'.format(digits)), rounding=ROUND_HALF_UP)

def DFDall(original_sequence, up_disturbance_rate,down_disturbance_rate):
    burst_len=[]
    current_packet=original_sequence[0]
    current_count=0
    disturbed_sequence=[]

    i=0
    inject_sum=0
    for packet in original_sequence:
        i=i+1
        disturbed_sequence.append(packet)
        if packet==0:
            break
        
        else:
            if packet==current_packet:
                current_count=current_count+1
                if(current_count==2 and len(burst_len)>1):
                    if(current_packet==1):
                        inject_num=int(round_up(burst_len[len(burst_len)-2]*up_disturbance_rate))
                    elif(current_packet==-1):
                        inject_num=int(round_up(burst_len[len(burst_len)-2]*down_disturbance_rate))
                    inject_sum+=inject_num
                    disturbance_injection = [current_packet] * inject_num  # 插入的数据包，假设插入的包的内容是 '1'，可以根据需要调整
                    disturbed_sequence.extend(disturbance_injection)
            else:
                
                burst_len.append(current_count)
                current_packet=packet
                current_count=1
    return disturbed_sequence,inject_sum

def DFD_def(X_test,up_disturbance_rate,down_disturbance_rate):
    disturbed_X=[]
    inject_sum=0
    for sequence in tqdm(X_test):
        disturbed_sequence,injectsum_this= DFDall(sequence, up_disturbance_rate,down_disturbance_rate)  
        inject_sum+=injectsum_this
        if len(disturbed_sequence) < 200:
            disturbed_sequence.extend([0] * (200 - len(disturbed_sequence)))
        disturbed_X.append(disturbed_sequence[:200])  
    disturbed_X = np.array(disturbed_X,dtype=np.float64)
    return disturbed_X