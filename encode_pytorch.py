import torch

def global_to_relative(indices, H_in, W_in):
    """
    将全局索引转换为相对索引
    :param indices: (C, H_out, W_out) 形状的全局索引
    :param H_in: 原始输入的高度
    :param W_in: 原始输入的宽度
    :return: 相对索引 (C, H_out, W_out)，值范围 {0,1,2,3}
    """
    C, H_out, W_out = indices.shape

    # 计算全局坐标 (h_global, w_global)
    h_global = indices // W_in
    w_global = indices % W_in

    # 计算池化窗口左上角 (h0, w0)
    h0 = torch.arange(H_out).repeat(W_out, 1).T * 2  # (H_out, W_out)
    w0 = torch.arange(W_out).repeat(H_out, 1) * 2    # (H_out, W_out)
    h0, w0 = h0, w0

    # 计算相对坐标 (h_relative, w_relative)
    h_relative = h_global - h0.unsqueeze(0)  # (C, H_out, W_out)
    w_relative = w_global - w0.unsqueeze(0)  # (C, H_out, W_out)

    # 计算相对索引 (0,1,2,3)
    relative_index = h_relative * 2 + w_relative

    return relative_index


def relative_to_global(relative_index, H_in, W_in):
    """
    将相对索引恢复为全局索引
    :param relative_index: (C, H_out, W_out) 形状的相对索引，值范围 {0,1,2,3}
    :param H_in: 原始输入的高度
    :param W_in: 原始输入的宽度
    :return: (C, H_out, W_out) 形状的全局索引
    """
    C, H_out, W_out = relative_index.shape

    # 计算池化窗口左上角 (h0, w0)
    h0 = torch.arange(H_out).repeat(W_out, 1).T * 2  # (H_out, W_out)
    w0 = torch.arange(W_out).repeat(H_out, 1) * 2    # (H_out, W_out)
    h0, w0 = h0.to(relative_index.device), w0.to(relative_index.device)

    # 计算恢复的 h, w
    h_rel = relative_index // 2  # 计算相对 h 位置
    w_rel = relative_index % 2   # 计算相对 w 位置

    h_global = h0.unsqueeze(0) + h_rel  # 计算全局 h
    w_global = w0.unsqueeze(0) + w_rel  # 计算全局 w

    # 计算恢复的全局索引
    global_index = h_global * W_in + w_global

    return global_index


# 测试代码
C, H_in, W_in = 512, 8, 8  # 原始输入尺寸
H_out, W_out = H_in // 2, W_in // 2  # 池化后尺寸

x = torch.randint(0,64,(512, 8, 8))
# 进行 MaxPool，返回索引
pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
y, indices = pool(x)  # indices 形状为 (512, 4, 4)

# 测试全局索引 -> 相对索引 -> 还原全局索引
relative_index = global_to_relative(indices, H_in, W_in)
restored_indices = relative_to_global(relative_index, H_in, W_in)

# 验证是否一致
print(torch.all(indices == restored_indices))  # True
