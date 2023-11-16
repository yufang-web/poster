import os
import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tabulate import tabulate
import multiprocess as mp
import logging
import pyBigWig
from mpl_toolkits.axes_grid1 import make_axes_locatable

index_name=["AG_TC","AG_AG","UC_AG","UC_TC"]
class Triplex_plot():
    def __init__(self,seq_RNA_name,seq_DNA_name,kmer,cutoff,bwfile,bed,denoise):
        self.seq_RNA=seq_RNA_name
        self.seq_DNA=seq_DNA_name
        self.kmer=kmer
        self.cutoff = cutoff
        self.bed=bed
        self.bwfile=bwfile
        self.cutoff=cutoff
        self.denoise=denoise
    def get_seq(self):
        f1 = open(self.seq_RNA)
        f2 = open(self.seq_DNA)
        seq_1 = {}
        for line in f1:
            if line.startswith('>'):
                name = line.replace('>', '').split()[0]
                seq_1[name] = ''
            else:
                seq_1[name] += line.replace('\n', '').strip()
        f1.close()
        seq_2={}
        for line in f2:
            if line.startswith('>'):
                name = line.replace('>', '').split()[0]
                seq_2[name] = ''
            else:
                seq_2[name] += line.replace('\n', '').strip()
        f2.close()
        return(seq_1,seq_2)
    def part_seq(self,seq_1,seq_2):
        seq_1=list(seq_1.values())[0]
        seq_2=list(seq_2.values())[0]
        short_seq_list1=[]
        short_seq_list2=[]
        short_1=[]
        short_2=[]
        for i in range(len(seq_1) - self.kmer + 1):
            short_seq_list1.append(seq_1[i:i+self.kmer])
            short_1.append(i)
        for j in range(len(seq_2) - self.kmer + 1):
            short_seq_list2.append(seq_2[j:j+self.kmer])
            short_2.append(j)
        return(short_seq_list1,short_seq_list2,short_1,short_2)
def get_score(short_seq1, short_seq2):
    score_purine_1 = 0
    score_purine_2 = 0
    score_py_1 = 0
    score_py_2 = 0        
    reversed_short_seq2 = ''.join(reversed(short_seq2))
    score_m_1 = {'A': {'A': 0, 'C': 0, 'G': 0, 'T': 1},
                    'C': {'A': 0, 'C': 0, 'G': 0, 'T': 0},
                    'G': {'A': 0, 'C': 1, 'G': 0, 'T': 0},
                    'T': {'A': 0, 'C': 0, 'G': 0, 'T': 0}}

    score_m_2 = {'A': {'A': 1, 'C': 0, 'G': 0, 'T': 0},
                    'C': {'A': 0, 'C': 0, 'G': 0, 'T': 0},
                    'G': {'A': 0, 'C': 0, 'G': 1, 'T': 0},
                    'T': {'A': 0, 'C': 0, 'G': 0, 'T': 0}}

    score_m_3 = {'A': {'A': 0, 'C': 0, 'G': 0, 'T': 0},
                    'C': {'A': 0, 'C': 0, 'G': 1, 'T': 0},
                    'G': {'A': 0, 'C': 0, 'G': 0, 'T': 0},
                    'T': {'A': 1, 'C': 0, 'G': 0, 'T': 0}}

    score_m_4 = {'A': {'A': 0, 'C': 0, 'G': 0, 'T': 0},
                    'C': {'A': 0, 'C': 1, 'G': 0, 'T': 0},
                    'G': {'A': 0, 'C': 0, 'G': 1, 'T': 0},
                    'T': {'A': 0, 'C': 0, 'G': 0, 'T': 1}}

    score_purine_1 = sum(score_m_1[short_seq1[i]][short_seq2[i]] for i in range(len(short_seq1)))
    score_purine_2 = sum(score_m_2[short_seq1[i]][reversed_short_seq2[i]] for i in range(len(short_seq1)))
    score_py_1 = sum(score_m_3[short_seq1[i]][short_seq2[i]] for i in range(len(short_seq1)))
    score_py_2 = sum(score_m_4[short_seq1[i]][reversed_short_seq2[i]] for i in range(len(short_seq1)))
    return (score_purine_1, score_purine_2, score_py_1, score_py_2)

def compute_score_parallel(i, j, short_seq_list1, short_seq_list2):  
    score_1, score_2, score_3, score_4 = get_score(short_seq_list1[i], short_seq_list2[j])
    i,j=i,j
    return (i, j, score_1, score_2, score_3, score_4)

def run_predict(example):
    seq_1, seq_2 = example.get_seq()[0], example.get_seq()[1]
    short_seq_list1, short_seq_list2 = example.part_seq(seq_1, seq_2)[0], example.part_seq(seq_1, seq_2)[1]
    rna_len, dna_len = len(short_seq_list1), len(short_seq_list2)
    score_matrix_1 = np.zeros((rna_len, dna_len), dtype=int)
    score_matrix_2 = np.zeros((rna_len, dna_len), dtype=int)
    score_matrix_3 = np.zeros((rna_len, dna_len), dtype=int)
    score_matrix_4 = np.zeros((rna_len, dna_len), dtype=int)
    results=[]
    with mp.Pool(10) as pool:
        arg_list = [(i, j, short_seq_list1, short_seq_list2) for i in range(rna_len) for j in range(dna_len)]
        results = pool.starmap(compute_score_parallel, arg_list)
    for score in results:
        i, j, score_1, score_2, score_3, score_4 = score
        score_matrix_1[i, j] = score_1
        score_matrix_2[i, j] = score_2
        score_matrix_3[i, j] = score_3
        score_matrix_4[i, j] = score_4   
    return score_matrix_1, score_matrix_2, score_matrix_3, score_matrix_4, short_seq_list1, short_seq_list2

def find_continuous_diagonals(matrix, matrix_name):
    diagonals = []
    row_count, col_count = matrix.shape
    # Check diagonals from top-left to bottom-right
    for i in range(row_count):
        for j in range(col_count):
            diagonal = []
            r = i
            c = j
            while r < row_count and c < col_count and matrix[r][c] > 10:
                diagonal.append(matrix[r][c])
                r += 1
                c += 1
            if len(diagonal) >= 2:
                diagonals.append({
                    'diagonal': diagonal,
                    'row_range': (i, r - 1),
                    'col_range': (j, c - 1),
                    'sum': sum(diagonal),
                    'max_value': max(diagonal),
                    'matrix': matrix_name
                })
    # Check diagonals from top-right to bottom-left
    for i in range(row_count):
        for j in range(col_count - 1, -1, -1):
            diagonal = []
            r = i
            c = j
            while r < row_count and c >= 0 and matrix[r][c] > 10:
                diagonal.append(matrix[r][c])
                r += 1
                c -= 1
            if len(diagonal) >= 2:
                diagonals.append({
                    'diagonal': diagonal,
                    'row_range': (i, r - 1),
                    'col_range': (j, c + 1),
                    'sum': sum(diagonal),
                    'max_value': max(diagonal),
                    'matrix': matrix_name
                })
    return pd.DataFrame(diagonals)
# Define a function to process a list of matrices

def process_matrices(matrices):
    # Process each matrix and store the results in a list
    results = []
    for i, matrix in enumerate(matrices):
        matrix_name = index_name[i]
        # Find continuous diagonals and create a DataFrame
        result_df = find_continuous_diagonals(matrix, matrix_name)
        results.append(result_df)
    # Combine all results into a single DataFrame
    combined_results = pd.concat(results)
    # Sort the DataFrame based on maximum value and sum
    combined_results = combined_results.sort_values(['max_value', 'sum'], ascending=False)
    # Reset the index of the DataFrame
    combined_results.reset_index(drop=True, inplace=True)
    return combined_results

def plot_heatmap_with_bigwig(heatmap_data, bigwig_file, chromosome, start, end, output_file):
    # Calculate the width of the Bigwig region
    region_width = end - start

    # Adjust the width of the heatmap to match the Bigwig region
    adjusted_heatmap_data = heatmap_data[:, :region_width]

    # Create subplots
    fig, (ax_heatmap, ax_signal) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 0.2]})

    # Plot the heatmap
    heatmap_img = ax_heatmap.imshow(adjusted_heatmap_data, cmap='hot_r', aspect='auto')
    divider = make_axes_locatable(ax_heatmap)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(heatmap_img, cax=cax, orientation='vertical')
    ax_heatmap.set_title('Heatmap')

    # Read Bigwig data
    bw = pyBigWig.open(bigwig_file)
    signal = bw.values(chromosome, start, end)

    # Plot the signal
    ax_signal.plot(signal)
    ax_signal.set_title('Signal')
    ax_signal.set_xlabel('Position')
    ax_signal.set_ylabel('Value')

    plt.subplots_adjust(hspace=0.5)

    # Set the titles of the subplots based on chromosome and start/end positions
    ax_heatmap.set_title(f'Heatmap - {chromosome}:{start}-{end}')
    ax_signal.set_title(f'Signal - {chromosome}:{start}-{end}')

    # Save the image as a PNG file
    plt.savefig(output_file, format='png', dpi=3000)
    plt.close()

def read_bed_file(bed_file):
    with open(bed_file, 'r') as file:
        line = file.readline().strip()  # Read the first line
        fields = line.split('\t')  # Split the line by tab delimiter
        chromosome = fields[0]
        start = int(fields[1])
        end = int(fields[2])
    return chromosome, start, end

def get_option():
    parse = argparse.ArgumentParser(description='plot triplex for RNA and DNA data')
    parse.add_argument('-RNA', type=str, help='RNA file name')
    parse.add_argument('-DNA', type=str, help='DNA file name')
    parse.add_argument('-kmer', type=int, help='DNA kmer')
    parse.add_argument('--bwfile',type=str,default='No value',help='integration of Chip-seq signal file')
    parse.add_argument('--cutoff',type=float,default=0.8,help='the percentage of passing filter of score of Triple helics')
    parse.add_argument('--bed',type=str,default='No value',help='bed file for plotting the chip-signal')
    parse.add_argument('--denoise',type=str,default='FALSE',help='whether to denoise')
    args = parse.parse_args()
    return args

if __name__=="__main__":
    args=get_option()
    Triplex_example=Triplex_plot(args.RNA,args.DNA,args.kmer,args.cutoff,args.denoise,args.bed,args.bwfile)
    matrix_1,matrix_2,matrix_3,matrix_4=run_predict(Triplex_example)[:4]
    list1,list2=run_predict(Triplex_example)[4],run_predict(Triplex_example)[5]
    ###initiate pairs
    cutoff=args.cutoff
    kmer = args.kmer
    def zero_below_five(matrix):               
        mean_value = np.mean(matrix)   
        matrix[matrix<5]= 0
        matrix[(matrix > 4) & (matrix < 10)] = 1   
        return matrix  
    if args.denoise =="FALSE":
        pass
    elif args.denoise=="TRUE":
        matrix_1,matrix_2,matrix_3,matrix_4=zero_below_five(matrix_1),zero_below_five(matrix_2),zero_below_five(matrix_3),zero_below_five(matrix_4)
    #do filter for pairs within all pairs
    matrices = [matrix_1, matrix_2, matrix_3, matrix_4]
    ##function for filter
    def process_matrix_parallel(matrix, list1, list2, kmer, cutoff):
        result = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j]/kmer >= cutoff:
                    result.append([i,j,matrix[i,j],list1[i],list2[j]])
        return result
    ###process matrix with multiprocessing
    def transform_matrix_parallel(matrices, list1, list2, kmer, cutoff):
        with mp.Pool(10) as pool:
            args_list=[(matrix, list1, list2, kmer, cutoff) for matrix in matrices]
            results = pool.starmap(process_matrix_parallel, args_list)        
        return results
    result = transform_matrix_parallel(matrices, list1, list2, kmer, cutoff)
    ##whether to log transform data
    with open("my_file.txt", "w") as f:
        for row in result:
            for text in row:
              f.write(str(text))
              f.write(' ')
              f.write("\n")    
    np.savetxt('matrix1.txt', matrix_1, fmt='%d')
    np.savetxt('matrix2.txt', matrix_2, fmt='%d')
    np.savetxt('matrix3.txt', matrix_3, fmt='%d')
    np.savetxt('matrix4.txt', matrix_4, fmt='%d')
    combined_results = process_matrices(matrices)
    # Save the combined results to an Excel file
    combined_results.to_excel('diagonal_results.xlsx', index=False)
    bigwig_file=args.bwfile
    output_file="out.png"
    if args.bed == 'No value':
        for i, matrix in enumerate(matrices):
            plt.figure(dpi=5000)
            np.set_printoptions(suppress=True)         
            plt.imshow(matrix, cmap=plt.cm.hot_r, aspect='auto')
            plt.colorbar()
            plt.savefig(f'./matrix{i+1}.png', dpi=5000)
    else:
        chromosome,start,end='chr11',5251058,5251857  
        for i, matrix in enumerate(matrices):
            output_file = f'{index_name[i]}_heatmap.png'
            plot_heatmap_with_bigwig(matrix, bigwig_file, chromosome, start, end, output_file)
        

