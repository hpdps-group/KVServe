import json
import argparse


def calculate_latency(original_size_mb, cr, bandwidth_gbps, compression_overhead_sec):
    """
    计算延迟
    
    参数:
    - original_size_mb: 原始数据大小（MB）
    - cr: 压缩率
    - bandwidth_gbps: 网络带宽（Gbps）
    - compression_overhead_sec: 压缩和解压开销（秒）
    
    返回:
    - 总延迟（秒）
    """
    # 单位换算
    # 1 MB = 10^6 bytes
    # 1 Gbps = 10^9 bits/s = 10^9 / 8 bytes/s
    original_size_bytes = original_size_mb * 1024 * 1024
    bandwidth_bytes_per_sec = bandwidth_gbps * 1024 * 1024 * 1024 / 8
    
    # 计算压缩后的数据大小（bytes）
    compressed_size_bytes = original_size_bytes / cr
    
    # 计算网络传输时间（秒）
    network_transmission_time = compressed_size_bytes / bandwidth_bytes_per_sec
    
    # 总延迟 = 网络传输时间 + 压缩解压开销
    total_latency = network_transmission_time + compression_overhead_sec
    
    return total_latency * 1000 # ms


def update_latency_in_json(json_file_path, bandwidth_gbps, original_size_mb, compression_overhead_sec):
    """
    读取JSON文件，更新latency字段，并写回文件
    
    参数:
    - json_file_path: JSON文件路径
    - bandwidth_gbps: 网络带宽（Gbps）
    - original_size_mb: 原始数据大小（MB）
    - compression_overhead_sec: 压缩和解压开销（秒）
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 更新每条记录的latency字段
    for record in data:
        if 'cr' in record:
            cr = record['cr']
            new_latency = calculate_latency(original_size_mb, cr, bandwidth_gbps, compression_overhead_sec)
            record['latency'] = new_latency
            print(f"Config ID {record.get('config_id', 'N/A')}: CR={cr:.4f}, New Latency={new_latency:.4f}ms")
    
    # 写回JSON文件
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"\n已成功更新 {len(data)} 条记录的latency字段并保存到 {json_file_path}")


def main():
    parser = argparse.ArgumentParser(description='更新JSON文件中的latency字段')
    parser.add_argument('--json_file', type=str, default='none_merged_qasper_bitcomp.json',
                        help='JSON文件路径（默认: hadamard_merged_qasper.json）')
    parser.add_argument('--bandwidth', type=float, default=100,
                        help='网络带宽（Gbps），例如: 100')
    parser.add_argument('--data_size', type=float, default=2024.77,
                        help='原始数据大小（MB），例如: 2024')
    parser.add_argument('--compression_overhead', type=float, default=0.0431,
                        help='压缩和解压开销（秒），例如: 0.5')
    
    args = parser.parse_args()
    
    print(f"参数设置:")
    print(f"  网络带宽: {args.bandwidth} Gbps")
    print(f"  原始数据大小: {args.data_size} MB")
    print(f"  压缩/解压开销: {args.compression_overhead} 秒")
    print(f"  处理文件: {args.json_file}\n")
    
    update_latency_in_json(
        args.json_file,
        args.bandwidth,
        args.data_size,
        args.compression_overhead
    )


if __name__ == '__main__':
    main()

