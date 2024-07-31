def transpose(matrix, rows, cols):
    transposed_matrix = [[0 for _ in range(rows)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix
rows = int(input("Enter the no. rows of the matrix: "))
cols = int(input("Enter the no. columns of the matrix: "))
matrix = []

print("Enter the elements of the matrix: ")
for i in range(rows):
    a = []
    for j in range(cols):
        a.append(int(input(f"Element [{i+1}][{j+1}]: ")))
    matrix.append(a)

# Transpose the matrix
transposed = transpose(matrix, rows, cols)
print("\nTransposed matrix:")
for row in transposed:
 print(row)'''