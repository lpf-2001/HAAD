
import numpy as np
def get_pertubation(CFmodel_name,Dataset_name):
    if Dataset_name=='sirinam95':
        if CFmodel_name=='ensemble':
            noise=[[12, 6], [6, 6], [34, 6], [40, 6], [16, 6]]
        elif CFmodel_name=='df':
            noise=[[25, 6], [11, 6], [8, 6], [16, 6], [18, 6]]
    elif Dataset_name=='rimmer100':
        if CFmodel_name=='ensemble':
            noise=[[3, 6], [15, 6], [31, 6], [27, 6], [19, 6]]
        elif CFmodel_name=='df':
            noise=[[3, 6], [17, 6], [32, 6], [36, 6], [42, 6]]
        
    else:
        raise TypeError("No such Dataset_name")
    noise=np.array(noise, dtype=np.float32)
    return noise


    # # #trace shape [Ncity,2], the shape of old_trace is [batch_size,200,1],返回值[batch_size,200,1]
def generate_adv_trace(trace, old_trace):

    # 提前检查 trace 是否为空
    # if trace.numel() == 0:
    if trace.size == 0:
        return old_trace

    # 在 CPU 上对插入位置排序
    insert_loc = np.argsort(trace[:, 0])
    adv_trace = trace[insert_loc]

    # 向量化插入操作
    # insert_positions = adv_trace[:, 0].long()
    # insert_counts = adv_trace[:, 1].long()
    insert_positions = adv_trace[:, 0].astype(np.int64)
    insert_counts = adv_trace[:, 1].astype(np.int64)

    
    # 计算总插入偏移
    total_insert = insert_counts.sum().item()
    if total_insert == 0:
        return old_trace

    # 计算新 trace 的 shape
    batch_size, seq_len, _ = old_trace.shape
    new_seq_len = seq_len + total_insert
    new_trace = np.zeros((batch_size, new_seq_len, 1), dtype=np.float32)
    prev_pos = 0
    pos_offset = 0

    # 逐步插入
    for i in range(len(insert_positions)):
        pos = insert_positions[i].item()
        insert_num = insert_counts[i].item()
        # 复制当前 segment
        new_trace[:, prev_pos+pos_offset:pos+pos_offset, :] = old_trace[:, prev_pos:pos, :]

        # 插入元素
        if i == 0:
            insert_val = old_trace[:, pos, :]
        else:
            insert_val = new_trace[:, pos, :]
        
        # new_trace[:,pos+pos_offset: pos+pos_offset + insert_num, :] = insert_val.unsqueeze(1).expand(-1, insert_num, -1)
        # 先扩展 insert_val，复制 insert_num 次
        expanded_insert_val = np.repeat(insert_val[:, np.newaxis, :], insert_num, axis=1)

        # 赋值到 new_trace 对应区域
        new_trace[:, pos+pos_offset : pos+pos_offset + insert_num, :] = expanded_insert_val

        prev_pos = pos 
        pos_offset += insert_num
    # 复制剩余部分
    new_trace[:, prev_pos+pos_offset:, :] = old_trace[:, prev_pos:, :]
    return new_trace[:,:seq_len,:]


def HAAD_pertubation(CFmodel_name,Dataset_name,data):
    noise=get_pertubation(CFmodel_name,Dataset_name)
    perturbed_data = generate_adv_trace(noise,data)
    return perturbed_data