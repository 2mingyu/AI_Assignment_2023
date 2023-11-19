// ..\mnist_dataset ������ �н�/�׽�Ʈ csv ���� �̿�
// ..\dataset\dataset.ipynb �� ���� �ձ۾� �н�/�׽�Ʈ csv ���� �̿�

// MNIST �н� ������ 1000��, EMNIST �н� ������ 1000�� ����
// // ��ü ����ð� �� 20�� (ȯ�濡 ���� �ٸ� �� ����)
// MNIST �׽�Ʈ 13/20, EMNIST �׽�Ʈ 15/20
// 
// MNIST �н� ������ 10000��, EMNIST �н� ������ 10000�� ����
// ��ü ����ð� �� 6�� 30�� (ȯ�濡 ���� �ٸ� �� ����)
// MNIST �׽�Ʈ 18/20, EMNIST �׽�Ʈ 15/20

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ����ġ ��� �ʱ�ȭ �Լ�
double** initialize_weights(int inputnodes, int outputnodes) {
    double** weights = (double**)malloc(outputnodes * sizeof(double*));
    for (int i = 0; i < outputnodes; i++) {
        weights[i] = (double*)malloc(inputnodes * sizeof(double));
        for (int j = 0; j < inputnodes; j++) {
            weights[i][j] = (double)rand() / RAND_MAX - 0.5;
        }
    }
    return weights;
}

// �ñ׸��̵� �Լ�
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Ȱ��ȭ �Լ�
// ����� �� ��ҿ� �ñ׸��̵� �Լ� ����
double** activation_function(double** a, int rows, int cols) {
    double** result = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        result[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            result[i][j] = sigmoid(a[i][j]);
        }
    }
    return result;
}


// 1���� �迭�� 2���� ��ķ� ��ȯ
double** convert_to_matrix(double* a, int rows) {
    double** result = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        result[i] = (double*)malloc(sizeof(double));
        result[i][0] = a[i];
    }
    return result;
}

// ���+��� ��� �Լ�
double** matrix_add(double** a, double** b, int rows, int cols) {
    double** result = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        result[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

// ���-��� ��� �Լ�
double** matrix_sub(double** a, double** b, int rows, int cols) {
    double** result = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        result[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    return result;
}

// ���*��� ��� �Լ�
double** matrix_mul(double** a, double** b, int rows, int cols) {
    double** result = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        result[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            result[i][j] = a[i][j] * b[i][j];
        }
    }
    return result;
}

// ��İ� ��� �Լ�
double** matrix_dot(double** a, double** b, int rows_a, int cols_a, int cols_b) {
    // cols_a = rows_b
    double** result = (double**)malloc(rows_a * sizeof(double*));
    for (int i = 0; i < rows_a; i++) {
        result[i] = (double*)malloc(cols_b * sizeof(double));
        for (int j = 0; j < cols_b; j++) {
            result[i][j] = 0;
            for (int k = 0; k < cols_a; k++) {
                result[i][j] = result[i][j] + (a[i][k] * b[k][j]);
            }
        }
    }
    return result;
}

// ��Į��-��� ��� �Լ�
double** scalar_matrix_sub(double s, double** a, int rows, int cols) {
    double** result = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        result[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            result[i][j] = s - a[i][j];
        }
    }
    return result;
}

// ��Į��*��� ��� �Լ�
double** scalar_matrix_mul(double s, double** a, int rows, int cols) {
    double** result = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        result[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            result[i][j] = s * a[i][j];
        }
    }
    return result;
}

// ��� ��ġ �Լ�
double** matrix_transpose(double** a, int rows, int cols) {
    double** result = (double**)malloc(cols * sizeof(double*));
    for (int i = 0; i < cols; i++) {
        result[i] = (double*)malloc(rows * sizeof(double));
        for (int j = 0; j < rows; j++) {
            result[i][j] = a[j][i];
        }
    }
    return result;
}



// �Ű�� ����ü ����
typedef struct {
    int inodes;
    int hnodes;
    int onodes;
    double** wih;
    double** who;
    double lr;
} NeuralNetwork;

// �Ű�� �ʱ�ȭ �Լ�
NeuralNetwork* initialize_neural_network(int inputnodes, int hiddennodes, int outputnodes, double learningrate) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->inodes = inputnodes;
    nn->hnodes = hiddennodes;
    nn->onodes = outputnodes;
    nn->wih = initialize_weights(inputnodes, hiddennodes);  // rows=hnodes, cols=inodes
    nn->who = initialize_weights(hiddennodes, outputnodes); // rows=onodes, cols=hnodes
    nn->lr = learningrate;
    return nn;
}

// �Ű�� �н� �Լ�
void train(NeuralNetwork* nn, double* inputs_list, double* targets_list) {
    // �Է� ����Ʈ, target ����Ʈ�� 2������ ��ķ� ��ȯ
    double** inputs = convert_to_matrix(inputs_list, nn->inodes);   // rows=inodes, cols=1
    double** targets = convert_to_matrix(targets_list, nn->onodes); // rows=onodes, cols=1
    
    // ���� �������� ������ ��ȣ ���
    double** hidden_inputs = matrix_dot(nn->wih, inputs, nn->hnodes, nn->inodes, 1); // row�� hnodes, cols�� 1
    // ���� �������� ������ ��ȣ ���
    double** hidden_outputs = activation_function(hidden_inputs, nn->hnodes, 1); // rows=hnodes, cols=1
    // ��� �������� ������ ��ȣ ���
    double** final_inputs = matrix_dot(nn->who, hidden_outputs, nn->onodes, nn->hnodes, 1); // rows=onodes, cols=1
    // ��� �������� ������ ��ȣ ���
    double** final_outputs = activation_function(final_inputs, nn->onodes, 1); // rows=onodes, cols=1
    // ��� ������ ���� ���
    double** output_errors = matrix_sub(targets, final_outputs, nn->onodes, 1); // rows=onodes, cols=1
    // ���� ������ ���� ���
    double** transposed_who = matrix_transpose(nn->who, nn->onodes, nn->hnodes); // rows=hnodes, cols=onodes
    double** hidden_errors = matrix_dot(transposed_who, output_errors, nn->hnodes, nn->onodes, 1); // rows=hnodes, cols=1   // ���� ���� ����
    // ���� ������ ��� ���� ���� ����ġ ��� ������Ʈ
    // ���̽� : self.who = matrix_add(self.who, scalar_matrix_mul(self.lr, matrix_dot((matrix_mul(matrix_mul(output_errors, final_outputs), scalar_matrix_sub(1, final_outputs))), matrix_transpose(hidden_outputs))))
    double** tmp15 = matrix_mul(output_errors, final_outputs, nn->onodes, 1); // rows=onodes, cols=1
    double** tmp16 = scalar_matrix_sub(1, final_outputs, nn->onodes, 1); // rows=onodes, cols=1
    double** tmp13 = matrix_mul(tmp15, tmp16, nn->onodes, 1); // rows=onodes, cols=1
    double** tmp14 = matrix_transpose(hidden_outputs, nn->hnodes, 1); // rows=1, cols=hnodes
    double** tmp12 = matrix_dot(tmp13, tmp14, nn->onodes, 1, nn->hnodes);   // rows=onodes, cols=hnodes
    double** tmp11 = scalar_matrix_mul(nn->lr, tmp12, nn->onodes, nn->hnodes); // rows=onodes, cols=hnodes
    double** old_who = nn->who;  // rows=onodes, cols=hnodes
    nn->who = matrix_add(old_who, tmp11, nn->onodes, nn->hnodes);
    // �Է� ������ ��� ���� ���� ����ġ ��� ������Ʈ
    // ���̽� : self.wih = matrix_add(self.wih, scalar_matrix_mul(self.lr, matrix_dot((matrix_mul(matrix_mul(hidden_errors, hidden_outputs), scalar_matrix_sub(1, hidden_outputs))), matrix_transpose(inputs))))
    double** tmp25 = matrix_mul(hidden_errors, hidden_outputs, nn->hnodes, 1); // rows=hnodes, cols=1
    double** tmp26 = scalar_matrix_sub(1, hidden_outputs, nn->hnodes, 1); // rows=hnodes, cols=1
    double** tmp23 = matrix_mul(tmp25, tmp26, nn->hnodes, 1); // rows=hnodes, cols=1
    double** tmp24 = matrix_transpose(inputs, nn->inodes, 1); // rows=1, cols=inodes
    double** tmp22 = matrix_dot(tmp23, tmp24, nn->hnodes, 1, nn->inodes);   // rows=hnodes, cols=inodes
    double** tmp21 = scalar_matrix_mul(nn->lr, tmp22, nn->hnodes, nn->inodes); // rows=hnodes, cols=inodes
    double** old_wih = nn->wih;  // rows=hnodes cols=indoes
    nn->wih = matrix_add(old_wih, tmp21, nn->hnodes, nn->inodes);
    // free
    for (int i = 0; i < nn->inodes; i++) {
        free(inputs[i]);
    }
    free(inputs);
    for (int i = 0; i < nn->hnodes; i++) {
        free(hidden_inputs[i]);
        free(hidden_outputs[i]);
        free(transposed_who[i]);
        free(hidden_errors[i]);
        free(tmp25[i]); free(tmp26[i]); free(tmp23[i]); free(tmp22[i]); free(tmp21[i]);
        free(old_wih[i]);
    }
    free(hidden_inputs);
    free(hidden_outputs);
    free(transposed_who);
    free(hidden_errors);
    free(tmp25); free(tmp26); free(tmp23); free(tmp22); free(tmp21);
    free(old_wih);
    for (int i = 0; i < nn->onodes; i++) {
        free(targets[i]);
        free(final_inputs[i]);
        free(final_outputs[i]);
        free(output_errors[i]);
        free(tmp15[i]); free(tmp16[i]); free(tmp13[i]); free(tmp12[i]); free(tmp11[i]);
        free(old_who[i]);
    }
    free(targets);
    free(final_inputs);
    free(final_outputs);
    free(output_errors);
    free(tmp15); free(tmp16); free(tmp13); free(tmp12); free(tmp11);
    free(old_who);

    free(tmp14[0]); free(tmp24[0]);
    free(tmp14); free(tmp24);
}

// �Ű�� �׽�Ʈ �Լ�
double** query(NeuralNetwork* nn, double* inputs_list) {
    // �Է� ����Ʈ�� 2���� ��ķ� ��ȯ
    double** inputs = convert_to_matrix(inputs_list, nn->inodes);   // rows=inodes, cols=1
    // ���� �������� ������ ��ȣ ���
    double** hidden_inputs = matrix_dot(nn->wih, inputs, nn->hnodes, nn->inodes, 1); // row�� hnodes, cols�� 1
    // ���� �������� ������ ��ȣ ���
    double** hidden_outputs = activation_function(hidden_inputs, nn->hnodes, 1); // rows=hnodes, cols=1
    // ��� �������� ������ ��ȣ ���
    double** final_inputs = matrix_dot(nn->who, hidden_outputs, nn->onodes, nn->hnodes, 1); // rows=onodes, cols=1
    // ��� �������� ������ ��ȣ ���
    double** final_outputs = activation_function(final_inputs, nn->onodes, 1); // rows=onodes, cols=1
    // free
    for (int i = 0; i < nn->inodes; i++) {
        free(inputs[i]);
    }
    free(inputs);
    for (int i = 0; i < nn->hnodes; i++) {
        free(hidden_inputs[i]);
        free(hidden_outputs[i]);
    }
    free(hidden_inputs);
    free(hidden_outputs);
    for (int i = 0; i < nn->onodes; i++) {
        free(final_inputs[i]);
    }
    free(final_inputs);
    return final_outputs;
}

void MNIST() {
    // ============================== MNIST ==============================

    // �Է�, ����, ��� ����� ��
    int input_nodes = 28 * 28;
    int hidden_nodes = 200;
    int output_nodes = 10;
    // �н���
    double learning_rate = 0.1;
    // �Ű�� �ʱ�ȭ
    NeuralNetwork* nn = initialize_neural_network(input_nodes, hidden_nodes, output_nodes, learning_rate);


    // MNIST �Ű�� �н� �ϱ�
    FILE* train_data_file = fopen("mnist_train.csv", "r");
    if (train_data_file == NULL) {
        perror("Error opening file");
        return;
    }
    char line[28 * 28 * 5]; // ����� ũ���� �迭 �Ҵ�
    int epochs = 1;
    printf("MNIST TRAIN\n");
    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("epoch %d / %d\n", epoch + 1, epochs);
        int line_n = 0;
        // ���Ͽ��� �� �پ� �о �н� ����
        while (fgets(line, sizeof(line), train_data_file) != NULL) {
            if (line_n == 1000) { break; }
            line_n += 1;
            // label �����
            int label = line[0] - '0';
            // targets �����
            double targets[10];
            for (int i = 0; i < 10; i++) {
                if (i == label) { targets[i] = 0.99; }
                else { targets[i] = 0.01; }
            }
            // ipnuts �����
            char* token = strtok(line + 2, ",");   // �󺧰� , �� ĭ�� �Ѿ
            double inputs[28 * 28];
            for (int i = 0; i < 28 * 28; i++) {
                double value = atoi(token);
                inputs[i] = (value / 255 * 0.99) + 0.01;
                token = strtok(NULL, ",");
            }
            free(token);
            // train ����
            train(nn, inputs, targets);
        }
        // ���� �����͸� ó������ �ǵ����� (epoch�� �ٽ� �����ϱ� ����)
        fseek(train_data_file, 0, SEEK_SET);
    }
    // ���� �ݱ�
    fclose(train_data_file);


    // MNIST �Ű�� �׽�Ʈ �ϱ�
    FILE* test_data_file = fopen("mnist_test.csv", "r");
    if (test_data_file == NULL) {
        perror("Error opening file");
        return;
    }
    char test_line[28 * 28 * 5];  // ����� ũ���� �迭 �Ҵ�
    int correct = 0;
    int total = 0;
    // ���Ͽ��� �� �پ� �о �׽�Ʈ ����
    printf("MNIST TEST\nlabel\tanswer\n");
    int test_line_n = 0;
    while (fgets(test_line, sizeof(test_line), test_data_file) != NULL) {
        if (test_line_n == 20) { break; }
        test_line_n += 1;
        // label �����
        int label = test_line[0] - '0';
        // ipnuts �����
        char* token = strtok(test_line + 2, ",");   // �󺧰� , �� ĭ�� �Ѿ
        double inputs[28 * 28];
        for (int i = 0; i < 28 * 28; i++) {
            double value = atoi(token);
            inputs[i] = (value / 255 * 0.99) + 0.01;
            token = strtok(NULL, ",");
        }
        free(token);
        // query ����
        double** outputs = query(nn, inputs);
        // ���� Ȯ��
        int answer = 0;
        double tmp = 0;
        for (int i = 0;i < 10;i++) {
            if (outputs[i][0] > tmp) { answer = i; tmp = outputs[i][0]; }
            free(outputs[i]);
        }
        free(outputs);
        printf("%d\t%d\n", label, answer);
        if (label == answer) { correct += 1; }
        total += 1;
    }
    // ���� �ݱ�
    fclose(test_data_file);


    // ��Ȯ�� ���
    printf("correct / total : %d / %d\n", correct, total);

    // free
    for (int i = 0; i < nn->onodes; i++) { free(nn->who[i]); }
    free(nn->who);
    for (int i = 0; i < nn->hnodes; i++) { free(nn->wih[i]); }
    free(nn->wih);
    free(nn);
}

void handwrite() {
    // ============================== �ձ۾� ==============================

    // �Է�, ����, ��� ����� ��
    int input_nodes = 16 * 16;
    int hidden_nodes = 200;
    int output_nodes = 7;
    // �н���
    double learning_rate = 0.1;
    // �Ű�� �ʱ�ȭ
    NeuralNetwork* nn = initialize_neural_network(input_nodes, hidden_nodes, output_nodes, learning_rate);


    // �ձ۾� �Ű�� �н� �ϱ�
    FILE* train_data_file = fopen("train.csv", "r");
    if (train_data_file == NULL) {
        perror("Error opening file");
        return;
    }
    char line[16 * 16 * 5]; // ����� ũ���� �迭 �Ҵ�
    int epochs = 1;
    printf("�ձ۾� TRAIN\n");
    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("epoch %d / %d\n", epoch + 1, epochs);
        int line_n = 0;
        // ���Ͽ��� �� �پ� �о �н� ����
        while (fgets(line, sizeof(line), train_data_file) != NULL) {
            if (line_n == 1000) { break; }
            line_n += 1;
            // label �����
            char* tmps = (char*)malloc(3);
            strncpy(tmps, line, 2, 2);
            tmps[2] = '\0';
            int label = atoi(tmps)-20;
            free(tmps);
            // targets �����
            double targets[7];
            for (int i = 0; i < 7; i++) {
                if (i == label) { targets[i] = 0.99; }
                else { targets[i] = 0.01; }
            }
            // ipnuts �����
            char* token = strtok(line + 3, ",");   // ��(���ڸ�)�� , �� ĭ�� �Ѿ
            double inputs[16 * 16];
            for (int i = 0; i < 16 * 16; i++) {
                double value = atoi(token);
                inputs[i] = (value / 255 * 0.99) + 0.01;
                token = strtok(NULL, ",");
            }
            free(token);
            // train ����
            train(nn, inputs, targets);
        }
        // ���� �����͸� ó������ �ǵ����� (epoch�� �ٽ� �����ϱ� ����)
        fseek(train_data_file, 0, SEEK_SET);
    }
    // ���� �ݱ�
    fclose(train_data_file);


    // �ձ۾� �Ű�� �׽�Ʈ �ϱ�
    FILE* test_data_file = fopen("test.csv", "r");
    if (test_data_file == NULL) {
        perror("Error opening file");
        return;
    }
    char test_line[16 * 16 * 5];  // ����� ũ���� �迭 �Ҵ�
    int correct = 0;
    int total = 0;
    // ���Ͽ��� �� �پ� �о �׽�Ʈ ����
    printf("�ձ۾� TEST\nlabel\tanswer\n");
    while (fgets(test_line, sizeof(test_line), test_data_file) != NULL) {
        // label �����
        char* tmps = (char*)malloc(3);
        strncpy(tmps, test_line, 2, 2);
        tmps[2] = '\0';
        int label = atoi(tmps) - 20;
        free(tmps);
        // ipnuts �����
        char* token = strtok(test_line + 2, ",");   // ��(���ڸ�)�� , �� ĭ�� �Ѿ
        double inputs[16 * 16];
        for (int i = 0; i < 16 * 16; i++) {
            double value = atoi(token);
            inputs[i] = (value / 255 * 0.99) + 0.01;
            token = strtok(NULL, ",");
        }
        free(token);
        // query ����
        double** outputs = query(nn, inputs);
        // ���� Ȯ��
        int answer = 0;
        double tmp = 0;
        for (int i = 0;i < 7;i++) {
            if (outputs[i][0] > tmp) { answer = i; tmp = outputs[i][0]; }
            free(outputs[i]);
        }
        free(outputs);
        printf("%d\t%d\n", label, answer);
        if (label == answer) { correct += 1; }
        total += 1;
    }
    // ���� �ݱ�
    fclose(test_data_file);


    // ��Ȯ�� ���
    printf("correct / total : %d / %d\n", correct, total);

    // free
    for (int i = 0; i < nn->onodes; i++) { free(nn->who[i]); }
    free(nn->who);
    for (int i = 0; i < nn->hnodes; i++) { free(nn->wih[i]); }
    free(nn->wih);
    free(nn);
}


int main() {
    MNIST();
    handwrite();
    return 0;
}

/*
(���� 1000���� �н� ���)
MNIST TRAIN
epoch 1 / 1
MNIST TEST
label   answer
7       7
2       5
1       1
0       0
4       4
1       1
4       3
9       4
5       6
9       7
0       0
6       2
9       9
0       0
1       1
5       5
9       7
7       7
3       3
4       4
correct / total : 13 / 20
�ձ۾� TRAIN
epoch 1 / 1
�ձ۾� TEST
label   answer
0       6
0       0
0       0
1       2
1       1
2       2
2       2
2       2
3       3
3       3
3       0
4       0
4       4
4       4
5       5
5       5
5       5
6       6
6       6
6       5
correct / total : 15 / 20

(���� 10000���� �н� ���)
MNIST TRAIN
epoch 1 / 1
MNIST TEST
label   answer
7       7
2       2
1       1
0       0
4       4
1       1
4       4
9       9
5       6
9       9
0       0
6       8
9       9
0       0
1       1
5       5
9       9
7       7
3       3
4       4
correct / total : 18 / 20
�ձ۾� TRAIN
epoch 1 / 1
�ձ۾� TEST
label   answer
0       0
0       0
0       0
1       0
1       1
2       0
2       2
2       2
3       0
3       3
3       0
4       4
4       4
4       4
5       5
5       5
5       5
6       6
6       6
6       5
correct / total : 15 / 20
*/