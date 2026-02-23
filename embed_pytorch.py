import torch
from encode import global_to_relative, relative_to_global


def get_decimal_1_to_n(x, n):
    abs_x = torch.abs(x)  # 取绝对值
    decimal_part = ((abs_x * 10 ** n).long() % 10 ** n)  # 取小数点后 1-n 位
    return decimal_part


def get_th_decimal(matrix, n):
    return ((torch.abs(matrix) * 10 ** n).long() % 10)


def embed_data_batch(int_matrices, float_matrices, precision_per_int=10):
    num_integers = 1
    total_precision = precision_per_int ** num_integers
    embedded_matrices = []

    for int_matrix, float_matrix in zip(int_matrices, float_matrices):
        # int_matrix = global_to_relative(int_matrix, 8, 8)
        flat_floats = int_matrix / 100000
        A_part = get_decimal_1_to_n(float_matrix, 4) / 10000
        C_part = torch.abs(float_matrix) - get_decimal_1_to_n(float_matrix, 6) / 1000000
        embedded_matrix = torch.sign(float_matrix) * (A_part + flat_floats + C_part)
        embedded_matrices.append(embedded_matrix)

    return torch.stack(embedded_matrices)


def extract_data_batch(embedded_matrices, original_shape, precision_per_int=10):
    num_integers = 1
    total_precision = precision_per_int ** num_integers
    extracted_matrices = []

    for embedded_matrix in embedded_matrices:
        extracted_int_matrix = get_th_decimal(embedded_matrix, 5)
        # extracted_int_matrix = relative_to_global(extracted_int_matrix, 8, 8)
        extracted_matrices.append(extracted_int_matrix)

    return torch.stack(extracted_matrices)


# Test code
if __name__ == "__main__":

    ###测试索引映射
    # batch_size = 10
    # C, H_in, W_in = 512, 8, 8  # 原始输入尺寸
    # H_out, W_out = H_in // 2, W_in // 2  # 池化后尺寸
    # x = torch.randint(0, 64, (batch_size, 512, 8, 8))
    # pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    # y, indices = pool(x)  # indices 形状为 (batch_size, 512, 4, 4)
    #
    # float_matrices = torch.rand(batch_size, 512, 4, 4)
    #
    # # Embedding process
    # embedded_matrices = embed_data_batch(indices, float_matrices)
    # # Extraction process
    # extracted_int_matrices = extract_data_batch(embedded_matrices, indices.shape[1:])
    #
    # # print("Consistency check for each batch item:", all(
    # #     torch.equal(original, extracted)
    # #     for original, extracted in zip(indices, extracted_int_matrices)
    # # ))
    # consistency = True
    # for i, (original, extracted) in enumerate(zip(indices, extracted_int_matrices)):
    #     if not torch.equal(original, extracted):
    #         consistency = False
    #         print(f"Batch index {i} has discrepancies:")
    #         diff_indices = (original != extracted).nonzero(as_tuple=True)
    #         for idx in zip(*diff_indices):
    #             print(f"Index {idx}: Original = {original[idx]}, Extracted = {extracted[idx]}")
    #
    # print("Consistency check for each batch item:", consistency)

    ###测试不包括索引映射
    ###测试索引映射
    batch_size = 10

    indices = torch.randint(0, 10, (batch_size, 512, 4, 4))
    float_matrices = torch.rand(batch_size, 512, 4, 4)
    #
    # Embedding process
    embedded_matrices = embed_data_batch(indices, float_matrices)
    # Extraction process
    extracted_int_matrices = extract_data_batch(embedded_matrices, indices.shape[1:])
    #
    # # print("Consistency check for each batch item:", all(
    # #     torch.equal(original, extracted)
    # #     for original, extracted in zip(indices, extracted_int_matrices)
    # # ))
    consistency = True
    for i, (original, extracted) in enumerate(zip(indices, extracted_int_matrices)):
        if not torch.equal(original, extracted):
            consistency = False
            print(f"Batch index {i} has discrepancies:")
            diff_indices = (original != extracted).nonzero(as_tuple=True)
            for idx in zip(*diff_indices):
                print(f"Index {idx}: Original = {original[idx]}, Extracted = {extracted[idx]}")

    print("Consistency check for each batch item:", consistency)


