import time

# WAN(Wide area network) Bandwidth, unit: 使用单位: Mbps (1 MB/s = 8 Mbps); 带宽测试: 40Mbps (5MB/s)
WAN_bandwidth = 10 # Mbps
mem_occupancy = 8 # B 字节 
train_epoch = 1
# 计算时: 元素个数 * 4 B / 1024 / 1024 MB  / (40/8) s = object_num * self.mem_occupancy / (1024*1024) / (self.WAN_bandwidth/8)
def compute_time_secureML(n, d, B):
    share_data_ = n * d
    share_E_ = 2 * n * d
    comm_data = 2 * (B + d) * (n/B)

    total_object_num = share_data_+share_E_ + comm_data * train_epoch

    commTime = total_object_num * mem_occupancy / (1024*1024) / (WAN_bandwidth/8)
    
    print("secureML comm time: " + str(commTime/3600) + " h")

def compute_time_secureML2(n, d, B):
    share_data_ = n * d
    share_E_ = n * d
    comm_data = 4 * (n * d)

    total_object_num = share_data_+share_E_ + comm_data * train_epoch

    commTime = total_object_num * mem_occupancy / (1024*1024) / (WAN_bandwidth/8)
    
    print("secureML comm time: " + str(commTime/3600) + " h")

def compute_time_CAESAR():
    pass

if __name__ == "__main__":
    # compute_time_secureML(1e6, 1e5, 4096)
    # compute_time_secureML(5e3, 5e3, 4096)
    compute_time_secureML2(1e6, 1e5, 4096)
    compute_time_secureML2(5e3, 5e3, 4096)