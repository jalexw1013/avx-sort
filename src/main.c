#include <stdio.h>
#include <sys/time.h>
#include <stdint.h>
#include <float.h>
#include <getopt.h>
#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <malloc.h>
#include <x86intrin.h>
#include <unistd.h>
#include <inttypes.h>
#include <immintrin.h>

#include "utils/util.h"
#include "utils/xmalloc.h"
#include "sorts.h"
#include "main.h"

// Random Tuning Parameters
//////////////////////////////
#define INFINITY_VALUE 1073741824
#define NEGATIVE_INFINITY_VALUE 0

// Function Prototypes
//////////////////////////////
void hostParseArgs(
    int argc, char** argv);
void initArrays(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted);
void insertData(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted);
#ifdef MERGE
void mergeTester(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted, uint32_t runs);
#endif
#ifdef SORT
void sortTester(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted, uint32_t runs);
#endif
#ifdef PARALLELSORT
void parallelTester(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted, uint32_t runs);
#endif

void freeGlobalData();
#ifdef MERGE
void initMergeFilePointer(FILE** fp);
#endif
#ifdef SORT
void initSortFilePointer(FILE** fp);
#endif
#ifdef PARALLELSORT
void initParallelSortFilePointer(FILE** fp);
#endif


// Global Host Variables
////////////////////////////
vec_t*    globalA;
vec_t*    globalB;
vec_t*    globalC;
vec_t*    CSorted;
vec_t*    CUnsorted;
#ifdef MERGE
FILE *mergeFile;
#endif
#ifdef SORT
FILE *sortFile;
#endif
#ifdef PARALLELSORT
FILE *parallelSortFile;
#endif
uint32_t  h_ui_A_length                = 8388608;
uint32_t  h_ui_B_length                = 8388608;
uint32_t  h_ui_C_length                = 16777216; //array to put values in
uint32_t  h_ui_Ct_length               = 16777216; //for unsorted and sorted
uint32_t  RUNS                         = 1;
uint32_t  entropy                      = 28;
uint32_t  OutToFile                    = 0; // 1 if output to file

//These Variables for a full testing run
const uint32_t testingEntropies[] = {28};//{4,8,12,16,20,24,28};//{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40};
const uint32_t testingEntropiesLength = 1;
const uint32_t testingSizes[] = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000};//{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100};
const uint32_t testingSizesLength = 900;
const uint32_t testingThreads[] = {1};//{1,2,4,8,16,24,32,40,48,64,72,128,144,180,256,270};//*/{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100};
const uint32_t testingThreadsLength = 1;

// Host Functions
////////////////////////////
int main(int argc, char** argv)
{
    // parse langths of A and B if user entered
    hostParseArgs(argc, argv);

    uint32_t seed = time(0);
    srand(seed);

    #ifdef MERGE
    if (OutToFile) {
        initMergeFilePointer(&mergeFile);
    }
    #endif

    #ifdef SORT
    if (OutToFile) {
        initSortFilePointer(&sortFile);
    }
    #endif

    #ifdef PARALLELSORT
    if (OutToFile) {
        initParallelSortFilePointer(&parallelSortFile);
    }
    #endif

    if (!OutToFile) {
        omp_set_dynamic(0);
        omp_set_num_threads(64);
        initArrays(
            &globalA, h_ui_A_length,
            &globalB, h_ui_B_length,
            &globalC, h_ui_C_length,
            &CSorted, h_ui_Ct_length,
            &CUnsorted);
        insertData(
            &globalA, h_ui_A_length,
            &globalB, h_ui_B_length,
            &globalC, h_ui_C_length,
            &CSorted, h_ui_Ct_length,
            &CUnsorted);
        #ifdef MERGE
        mergeTester(
            &globalA, h_ui_A_length,
            &globalB, h_ui_B_length,
            &globalC, h_ui_C_length,
            &CSorted, h_ui_Ct_length,
            &CUnsorted, RUNS);
        #endif
        #ifdef SORT
        sortTester(
            &globalA, h_ui_A_length,
            &globalB, h_ui_B_length,
            &globalC, h_ui_C_length,
            &CSorted, h_ui_Ct_length,
            &CUnsorted, RUNS);
        #endif
        #ifdef PARALLELSORT
        parallelTester(
            &globalA, h_ui_A_length,
            &globalB, h_ui_B_length,
            &globalC, h_ui_C_length,
            &CSorted, h_ui_Ct_length,
            &CUnsorted, RUNS);
        #endif

        freeGlobalData();
    } else {
        omp_set_dynamic(0);
        for (uint32_t i = 0; i < testingSizesLength; i++) {
            initArrays(
                &globalA, testingSizes[i]/2,
                &globalB, testingSizes[i]/2 + testingSizes[i]%2,
                &globalC, testingSizes[i],
                &CSorted, testingSizes[i],
                &CUnsorted);
            for (uint32_t e = 0; e < testingEntropiesLength; e++) {
                entropy = testingEntropies[e];
                insertData(
                    &globalA, testingSizes[i]/2,
                    &globalB, testingSizes[i]/2 + testingSizes[i]%2,
                    &globalC, testingSizes[i],
                    &CSorted, testingSizes[i],
                    &CUnsorted);
                #ifdef MERGE
                mergeTester(
                    &globalA, testingSizes[i]/2,
                    &globalB, testingSizes[i]/2 + testingSizes[i]%2,
                    &globalC, testingSizes[i],
                    &CSorted, testingSizes[i],
                    &CUnsorted, RUNS);
                #endif
                #ifdef SORT
                sortTester(
                    &globalA, testingSizes[i]/2,
                    &globalB, testingSizes[i]/2 + testingSizes[i]%2,
                    &globalC, testingSizes[i],
                    &CSorted, testingSizes[i],
                    &CUnsorted, RUNS);
                #endif
                #ifdef PARALLELSORT
                for (uint32_t j = 0; j < testingThreadsLength; j++) {
                    omp_set_num_threads(testingThreads[j]);
                    parallelTester(
                        &globalA, testingSizes[i]/2,
                        &globalB, testingSizes[i]/2 + testingSizes[i]%2,
                        &globalC, testingSizes[i],
                        &CSorted, testingSizes[i],
                        &CUnsorted, RUNS);
                }
                #endif
            }
            freeGlobalData();
        }
    }


}

int hostBasicCompare(const void * a, const void * b) {
    return (int) (*(vec_t *)a - *(vec_t *)b);
}

/**
 * Allocates the arrays A,B,C,CSorted, and CUnsorted
 */
void initArrays(vec_t** A, uint32_t A_length,
          vec_t** B, uint32_t B_length,
          vec_t** C, uint32_t C_length,
          vec_t** CSorted, uint32_t Ct_length,
          vec_t** CUnsorted)
{
    (*A)  = (vec_t*) xmalloc((A_length  + 8) * (sizeof(vec_t)));
    (*B)  = (vec_t*) xmalloc((B_length  + 8) * (sizeof(vec_t)));
    (*C)  = (vec_t*) xmalloc((C_length  + 32) * (sizeof(vec_t)));
    (*CSorted) = (vec_t*) xmalloc((Ct_length + 32) * (sizeof(vec_t)));
    (*CUnsorted) = (vec_t*) xmalloc((Ct_length + 32) * (sizeof(vec_t)));
}

/**
 * inserts the data into the arrays A,B,CUnsorted, and CSorted.
 * This can be called over and over to re randomize the data
 */
void insertData(vec_t** A, uint32_t A_length,
                vec_t** B, uint32_t B_length,
                vec_t** C, uint32_t C_length,
                vec_t** CSorted, uint32_t Ct_length,
                vec_t** CUnsorted)
{
    for(uint32_t i = 0; i < A_length; ++i) {
        (*A)[i] = rand() % (1 << (entropy - 1));
        (*CUnsorted)[i] = (*A)[i];
    }

    for(uint32_t i = 0; i < B_length; ++i) {
        (*B)[i] = rand() % (1 << (entropy - 1));
        (*CUnsorted)[i+A_length] = (*B)[i];
    }

    qsort((*A), A_length, sizeof(vec_t), hostBasicCompare);
    qsort((*B), B_length, sizeof(vec_t), hostBasicCompare);

    for(int i = 0; i < 8; ++i) {
        (*A)[A_length + i] = INFINITY_VALUE; (*B)[B_length + i] = INFINITY_VALUE;
    }

    // reference 'CORRECT' results
    uint32_t ai = 0,bi = 0,ci = 0;
    while(ai < A_length && bi < B_length) {
        (*CSorted)[ci++] = (*A)[ai] < (*B)[bi] ? (*A)[ai++] : (*B)[bi++];
    }
    while(ai < A_length) (*CSorted)[ci++] = (*A)[ai++];
    while(bi < B_length) (*CSorted)[ci++] = (*B)[bi++];
}

void freeGlobalData() {
      free(globalA);
      free(globalB);
      free(globalC);
      free(CSorted);
      free(CUnsorted);
}

#ifdef MERGE
void initMergeFilePointer(FILE** fp) {
    char fileName[50];
    sprintf(fileName, "../results/MergeResults.%li.csv", time(0));
    (*fp) = fopen(fileName, "w+");
    fprintf((*fp), "Name,Entropy,A Size,B Size,Elements Per Second,Total Time");
}

void writeToMergeOut(const char* name, uint32_t entropy, uint32_t ASize, uint32_t BSize, float time) {
    fprintf(mergeFile, "\n%s,%i,%i,%i,%i,%.20f", name, entropy, ASize, BSize, (int)((float)(ASize + BSize)/time), time);
}
#endif

#ifdef SORT
void initSortFilePointer(FILE** fp) {
    char fileName[50];
    sprintf(fileName, "../results/SortResults.%li.csv", time(0));
    (*fp) = fopen(fileName, "w+");
    fprintf((*fp), "Name,Entropy,C Size,Elements Per Second,Total Time");
}

void writeToSortOut(const char* name, uint32_t entropy, uint32_t CSize, float time) {
    fprintf(sortFile, "\n%s,%i,%i,%i,%.20f", name, entropy, CSize, (int)((float)(CSize)/time), time);
}
#endif

#ifdef PARALLELSORT
void initParallelSortFilePointer(FILE** fp) {
    char fileName[50];
    sprintf(fileName, "../results/ParallelSortResults.%li.csv", time(0));
    (*fp) = fopen(fileName, "w+");
    fprintf((*fp), "Name,Entropy,C Size,Number of Threads,Elements Per Second,Total Time");
}

void writeToParallelSortOut(const char* name, uint32_t entropy, uint32_t CSize, uint32_t numThreads, float time) {
    fprintf(parallelSortFile, "\n%s,%i,%i,%i,%i,%.20f", name, entropy, CSize, numThreads, (int)((float)(CSize)/time), time);
}
#endif

int verifyOutput(vec_t* output, vec_t* sortedData, uint32_t length, const char* name) {
    for(uint32_t i = 0; i < length; i++) {
        if(output[i] != sortedData[i]) {
            printf(ANSI_COLOR_RED "    Error: %s Failed To Produce Correct Results.\n", name);
            printf("    Index:%d, Given Value:%d, Correct "
            "Value:%d, ArraySize: %i" ANSI_COLOR_RESET "\n", i, output[i], sortedData[i], length);
            return 0;
        }
    }
    return 1;
}

void clearArray(vec_t* array, uint32_t length) {
    for (uint32_t i = 0; i < length; i++) {
        array[i] = 0;
    }
}

template <void (*Merge)(vec_t*,uint32_t,vec_t*,uint32_t,vec_t*,uint32_t, struct memPointers*)>
float testMerge(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t runs,
    const char* algoName) {

    //create input copies in case the algorithm acciedently changes the input
    vec_t* ACopy = (vec_t*)xmalloc((A_length + 8) * sizeof(vec_t));
    memcpy(ACopy, (*A), A_length * sizeof(vec_t));

    vec_t* BCopy = (vec_t*)xmalloc((B_length + 8) * sizeof(vec_t));
    memcpy(BCopy, (*B), B_length * sizeof(vec_t));

    uint32_t* ASplitters = (uint32_t*)xcalloc(17, sizeof(uint32_t));
    uint32_t* BSplitters = (uint32_t*)xcalloc(17, sizeof(uint32_t));
    struct memPointers* pointers = (struct memPointers*)xcalloc(1, sizeof(struct memPointers));
    pointers->ASplitters = ASplitters;
    pointers->BSplitters = BSplitters;

    //clear out array just to be sure
    clearArray((*C), C_length);

    //setup timing mechanism
    float time = 0.0;

    //reset timer
    tic_reset();

    //perform actual merge
    Merge((*A), A_length, (*B), B_length, (*C), C_length, pointers);

    //get timing
    time = tic_total();

    //verify output is valid
    if (!verifyOutput((*C), (*CSorted), C_length, algoName)) {
        //time = -1.0;
    }

    //restore original values
    clearArray((*C), C_length);
    memcpy( (*A), ACopy, A_length * sizeof(vec_t));
    memcpy( (*B), BCopy, B_length * sizeof(vec_t));

    free(ACopy);
    free(BCopy);
    free(ASplitters);
    free(BSplitters);
    free(pointers);

    return time;
}

template <void (*ParallelMerge)(vec_t*,uint32_t,vec_t*,uint32_t,vec_t*,uint32_t, struct memPointers*)>
float testParallelMerge(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t runs,
    const char* algoName) {

    //create input copies in case the algorithm acciedently changes the input
    vec_t* ACopy = (vec_t*)xmalloc((A_length + 8) * sizeof(vec_t));
    memcpy(ACopy, (*A), A_length * sizeof(vec_t));

    vec_t* BCopy = (vec_t*)xmalloc((B_length + 8) * sizeof(vec_t));
    memcpy(BCopy, (*B), B_length * sizeof(vec_t));

    //Check how many threads are running
    uint32_t numberOfThreads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            numberOfThreads = omp_get_num_threads();
        }
    }

    //allocate Variables
    uint32_t* ASplitters = (uint32_t*)xcalloc((numberOfThreads + 1)*numberOfThreads, sizeof(uint32_t));
    uint32_t* BSplitters = (uint32_t*)xcalloc((numberOfThreads + 1)*numberOfThreads, sizeof(uint32_t));
    struct memPointers* pointers = (struct memPointers*)xcalloc(1, sizeof(struct memPointers));
    pointers->ASplitters = ASplitters;
    pointers->BSplitters = BSplitters;

    //clear out array just to be sure
    clearArray((*C), C_length);

    //setup timing mechanism
    float time = 0.0;

    //reset timer
    tic_reset();

    //perform actual merge
    ParallelMerge((*A), A_length, (*B), B_length, (*C), C_length, pointers);

    //get timing
    time = tic_total();

    //verify output is valid
    if (!verifyOutput((*C), (*CSorted), C_length, algoName)) {
        //time = -1.0;
    }

    //restore original values
    clearArray((*C), C_length);
    memcpy( (*A), ACopy, A_length * sizeof(vec_t));
    memcpy( (*B), BCopy, B_length * sizeof(vec_t));

    free(ACopy);
    free(BCopy);
    free(ASplitters);
    free(BSplitters);
    free(pointers);

    return time;
}

template <SortTemplate Sort>
float testSort(
    vec_t** CUnsorted, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    uint32_t runs, const uint32_t splitNumber,
    const char* algoName) {

    //setup timing mechanism
    float time = 0.0;

    //store old values
    vec_t* unsortedCopy = (vec_t*)xmalloc(Ct_length * sizeof(vec_t));
    memcpy(unsortedCopy, (*CUnsorted), Ct_length * sizeof(vec_t));

    //allocate Variables
    vec_t* C = (vec_t*)xcalloc((C_length + 32), sizeof(vec_t));
    uint32_t* ASplitters = (uint32_t*)xcalloc(17, sizeof(uint32_t));
    uint32_t* BSplitters = (uint32_t*)xcalloc(17, sizeof(uint32_t));
    struct memPointers* pointers = (struct memPointers*)xcalloc(1, sizeof(struct memPointers));
    pointers->ASplitters = ASplitters;
    pointers->BSplitters = BSplitters;

    //reset timer
    tic_reset();

    //perform actual sort
    Sort((*CUnsorted), C, C_length, splitNumber, pointers);

    //get timing
    time += tic_sincelast();

    //verify output is valid
    if (!verifyOutput((*CUnsorted), (*CSorted), C_length, algoName)) {
        //time = -1.0;
    }

    //restore original values
    memcpy((*CUnsorted), unsortedCopy, Ct_length * sizeof(vec_t));

    //deallocate variables
    free(C);
    free(unsortedCopy);
    free(ASplitters);
    free(BSplitters);
    free(pointers);

    return time;
}

template <ParallelSortTemplate ParallelSort>
float testParallelSort(
    vec_t** CUnsorted, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    uint32_t runs, const uint32_t splitNumber,
    const char* algoName) {

    //setup timing mechanism
    float time = 0.0;

    //store old values
    vec_t* unsortedCopy = (vec_t*)xmalloc(Ct_length * sizeof(vec_t));
    memcpy(unsortedCopy, (*CUnsorted), Ct_length * sizeof(vec_t));

    //Check how many threads are running
    uint32_t numberOfThreads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            numberOfThreads = omp_get_num_threads();
        }
    }

    //allocate Variables
    vec_t* C = (vec_t*)xcalloc((C_length + 32), sizeof(vec_t));
    uint32_t* ASplitters = (uint32_t*)xcalloc((numberOfThreads + 1)*numberOfThreads, sizeof(uint32_t));
    uint32_t* BSplitters = (uint32_t*)xcalloc((numberOfThreads + 1)*numberOfThreads, sizeof(uint32_t));
    uint32_t* arraySizes = (uint32_t*)xcalloc(numberOfThreads*numberOfThreads, sizeof(uint32_t));
    struct memPointers* pointers = (struct memPointers*)xcalloc(1, sizeof(struct memPointers));
    pointers->ASplitters = ASplitters;
    pointers->BSplitters = BSplitters;
    pointers->arraySizes = arraySizes;

    //reset timer
    tic_reset();

    //perform actual sort
    ParallelSort((*CUnsorted), C, C_length, splitNumber, pointers);

    //get timing
    time += tic_sincelast();

    //verify output is valid
    if (!verifyOutput((*CUnsorted), (*CSorted), C_length, algoName)) {
        //time = -1.0;
    }

    //restore original values
    memcpy((*CUnsorted), unsortedCopy, Ct_length * sizeof(vec_t));

    //deallocate variables
    free(C);
    free(unsortedCopy);
    free(ASplitters);
    free(BSplitters);
    free(arraySizes);
    free(pointers);

    return time;
}


void printfcomma(int n) {
    if (n < 0) {
        printf ("N/A");
        return;
    }
    if (n < 1000) {
        printf ("%d", n);
        return;
    }
    printfcomma (n/1000);
    printf (",%03d", n%1000);
}

#ifdef MERGE
void mergeTester(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted, uint32_t runs)
{
    uint32_t numberOfThreads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            numberOfThreads = omp_get_num_threads();
        }
    }

    if (!OutToFile) {
        printf("Parameters\n");
        printf("Entropy: %d\n", entropy);
        printf("Runs: %i\n", runs);
        printf("A Size: %d\n", A_length);
        printf("B Size: %d\n", B_length);
        printf("Number Of Threads: %d\n", numberOfThreads);
        printf("\n");
    }

    double serialMergeTime = 0.0;
    double serialMergeNoBranchTime = 0.0;
    double bitonicMergeRealTime = 0.0;
    #ifdef AVX512
    double avx512MergeTime = 0.0;
    double avx512ParallelMergeTime = 0.0;
    #endif

    for (uint32_t run = 0; run < RUNS; run++) {
        serialMergeTime += testMerge<serialMerge>(
            A, A_length, B, B_length,
            C, Ct_length, CSorted,
            runs, "Serial Merge");

        serialMergeNoBranchTime += testMerge<serialMergeNoBranch>(
            A, A_length, B, B_length,
            C, Ct_length, CSorted,
            runs, "Branchless Merge");

        bitonicMergeRealTime += testMerge<bitonicMergeReal>(
            A, A_length, B, B_length,
            C, Ct_length, CSorted,
            runs, "Bitonic Merge");

        #ifdef AVX512
        avx512MergeTime += testMerge<avx512Merge>(
            A, A_length, B, B_length,
            C, Ct_length, CSorted,
            runs, "AVX-512 Merge");

        /*avx512ParallelMergeTime += testMerge<avx512ParallelMerge>(
            A, A_length, B, B_length,
            C, Ct_length, CSorted,
            runs, "AVX-512 Parallel Merge");*/
        #endif

        insertData(
            A, A_length,
            B, B_length,
            C, C_length,
            CSorted, Ct_length,
            CUnsorted);
    }

    serialMergeTime /= RUNS;
    serialMergeNoBranchTime /= RUNS;
    bitonicMergeRealTime /= RUNS;
    #ifdef AVX512
    avx512MergeTime /= RUNS;
    avx512ParallelMergeTime /= RUNS;
    #endif

    if (OutToFile) {
        writeToMergeOut("Serial Merge", entropy, A_length, B_length, serialMergeTime);
        writeToMergeOut("Serial Merge Branchless", entropy, A_length, B_length, serialMergeNoBranchTime);
        writeToMergeOut("Bitonic Merge", entropy, A_length, B_length, bitonicMergeRealTime);
        #ifdef AVX512
        writeToMergeOut("AVX512 Merge", entropy, A_length, B_length, avx512MergeTime);
        writeToMergeOut("AVX512 Parallel Merge", entropy, A_length, B_length, avx512ParallelMergeTime);
        #endif
    } else {
        printf("Merging Results        :  Elements per Second\n");
        printf("Serial Merge           :     ");
        if (serialMergeTime > 0.0) {
            printfcomma((int)((float)Ct_length/serialMergeTime));printf("\n");
        } else if (serialMergeTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        printf("Serial Merge Branchless:     ");
        if (serialMergeNoBranchTime > 0.0) {
            printfcomma((int)((float)Ct_length/serialMergeNoBranchTime));
        } else if (serialMergeNoBranchTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        printf("Bitonic Merge          :     ");
        if (bitonicMergeRealTime > 0.0) {
            printfcomma((int)((float)Ct_length/bitonicMergeRealTime));
        } else if (bitonicMergeRealTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        #ifdef AVX512
        printf("AVX512 Merge           :     ");
        if (avx512MergeTime > 0.0) {
            printfcomma((int)((float)Ct_length/avx512MergeTime));
        } else if (avx512MergeTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        printf("AVX512 Parallel Merge   :     ");
        if (avx512ParallelMergeTime > 0.0) {
            printfcomma((int)((double)Ct_length/avx512ParallelMergeTime));
        } else if (avx512ParallelMergeTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        #endif
        printf("\n\n");

        printf("Serial Merge           :     %f\n", serialMergeTime);
        printf("Serial Merge Branchless:     %f\n", serialMergeNoBranchTime);
        printf("Bitonic Merge          :     %f\n", bitonicMergeRealTime);
        #ifdef AVX512
        printf("AVX512 Merge           :     %f\n", avx512MergeTime);
        printf("AVX512 Parallel Merge  :     %f\n", avx512ParallelMergeTime);
        #endif
    }
}
#endif

#ifdef SORT
void sortTester(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted, uint32_t runs)
{
    if (!OutToFile) {
        printf("Parameters\n");
        printf("Entropy: %d\n", entropy);
        printf("Runs: %i\n", runs);
        //printf("Array Size: %" PRIu32 "\n", C_length);
        printf("\n");
    }

    float quickSortTime = 0.0;
    float serialMergeSortTime = 0.0;
    float serialMergeNoBranchSortTime = 0.0;
    float bitonicMergeRealSortTime = 0.0;
    #ifdef AVX512
    float avx512SortNoMergePathSerialTime = 0.0;
    float avx512SortNoMergePathV2SerialTime = 0.0;
    float avx512SortNoMergePathBranchlessTime = 0.0;
    float avx512SortNoMergePathBitonicTime = 0.0;
    float avx512SortNoMergePathavxTime = 0.0;
    float avx512MergeSortTime = 0.0;
    #endif

    for (uint32_t run = 0; run < RUNS; run++) {
        if (quickSortTime >= 0.0) {
            quickSortTime += testSort<quickSort>(
                CUnsorted, C_length,
                CSorted, Ct_length,
                runs, 64, "Quick Sort");
        }

        if (serialMergeSortTime >= 0.0) {
            serialMergeSortTime += testSort<iterativeMergeSort<serialMerge>>(
                CUnsorted, C_length,
                CSorted, Ct_length,
                runs, 64, "Merge Sort Standard");
        }

        if (serialMergeNoBranchSortTime >= 0.0) {
            serialMergeNoBranchSortTime += testSort<iterativeMergeSort<serialMergeNoBranch>>(
                CUnsorted, C_length,
                CSorted, Ct_length,
                runs, 64, "Branch Avoiding Sort");
        }

        if (bitonicMergeRealSortTime >= 0.0) {
            bitonicMergeRealSortTime += testSort<iterativeMergeSort<bitonicMergeReal>>(
                CUnsorted, C_length,
                CSorted, Ct_length,
                runs, 64, "Bitonic Based Merge Sort");
        }

        #ifdef AVX512

        if (avx512SortNoMergePathSerialTime >= 0.0) {
            avx512SortNoMergePathSerialTime += testSort<avx512SortNoMergePath<serialMerge>>(
                CUnsorted, C_length,
                CSorted, Ct_length,
                runs, 64, "AVX-512 Sort Without Merge Path Serial");
        }

        if (avx512SortNoMergePathV2SerialTime >= 0.0) {
            avx512SortNoMergePathV2SerialTime += testSort<avx512SortNoMergePathV2<serialMerge>>(
                CUnsorted, C_length,
                CSorted, Ct_length,
                runs, 64, "AVX-512 Sort Without Merge Path V2 Serial");
        }

        // if (avx512SortNoMergePathBranchlessTime >= 0.0) {
        //     avx512SortNoMergePathBranchlessTime += testSort<avx512SortNoMergePath<serialMergeNoBranch>>(
        //         CUnsorted, C_length,
        //         CSorted, Ct_length,
        //         runs, 64, "AVX-512 Sort Without Merge Path Branchless");
        // }
        //
        // if (avx512SortNoMergePathBitonicTime >= 0.0) {
        //     avx512SortNoMergePathBitonicTime += testSort<avx512SortNoMergePath<bitonicMergeReal>>(
        //         CUnsorted, C_length,
        //         CSorted, Ct_length,
        //         runs, 64, "AVX-512 Sort Without Merge Path Bitonic");
        // }
        //
        // if (avx512SortNoMergePathavxTime >= 0.0) {
        //     avx512SortNoMergePathavxTime += testSort<avx512SortNoMergePath<avx512Merge>>(
        //         CUnsorted, C_length,
        //         CSorted, Ct_length,
        //         runs, 64, "AVX-512 Sort Without Merge Path AVX");
        // }

        if (avx512MergeSortTime >= 0.0) {
            avx512MergeSortTime += testSort<iterativeMergeSort<avx512Merge>>(
                CUnsorted, C_length,
                CSorted, Ct_length,
                runs, 64, "AVX-512 Merge Sort");
        }
        #endif

        insertData(
            A, A_length,
            B, B_length,
            C, C_length,
            CSorted, Ct_length,
            CUnsorted);
    }

    // quickSortTime /= RUNS;
    // serialMergeSortTime /= RUNS;
    // serialMergeNoBranchSortTime /= RUNS;
    // bitonicMergeRealSortTime /= RUNS;
    // #ifdef AVX512
    // avx512SortNoMergePathSerialTime /= RUNS;
    // avx512SortNoMergePathV2SerialTime /= RUNS;
    // avx512SortNoMergePathBranchlessTime /= RUNS;
    // avx512SortNoMergePathBitonicTime /= RUNS;
    // avx512SortNoMergePathavxTime /= RUNS;
    // avx512MergeSortTime; /= RUNS;
    // #endif

    if (OutToFile) {
        writeToSortOut("Quick Sort", entropy, C_length, quickSortTime);
        writeToSortOut("Merge Sort Standard", entropy, C_length, serialMergeSortTime);
        writeToSortOut("Branch Avoiding Sort", entropy, C_length, serialMergeNoBranchSortTime);
        writeToSortOut("Bitonic Based Merge Sort", entropy, C_length, bitonicMergeRealSortTime);
        #ifdef AVX512
        writeToSortOut("AVX-512 Merge Sort", entropy, C_length, avx512MergeSortTime);
        #endif
    }

    if (!OutToFile) {
        printf("Sorting Results                         :  Elements per Second\n");
        printf("Quick Sort                              :     ");
        if (quickSortTime > 0.0) {
            printfcomma((int)((float)Ct_length/quickSortTime));
        } else if (quickSortTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        printf("Merge Sort Standard                     :     ");
        if (serialMergeSortTime > 0.0) {
            printfcomma((int)((float)Ct_length/serialMergeSortTime));
        } else if (serialMergeSortTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        printf("Branch Avoiding Sort                    :     ");
        if (serialMergeNoBranchSortTime > 0.0) {
            printfcomma((int)((float)Ct_length/serialMergeNoBranchSortTime));
        } else if (serialMergeNoBranchSortTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        printf("Bitonic Based Merge Sort                :     ");
        if (bitonicMergeRealSortTime > 0.0) {
            printfcomma((int)((float)Ct_length/bitonicMergeRealSortTime));
        } else if (bitonicMergeRealSortTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        #ifdef AVX512
        printf("AVX-512 Sort Without Merge Path Serial   :     ");
        if (avx512SortNoMergePathSerialTime > 0.0) {
            printfcomma((int)((float)Ct_length/avx512SortNoMergePathSerialTime));
        } else if (avx512SortNoMergePathSerialTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        printf("AVX-512 Sort Without Merge Path V2 Serial :     ");
        if (avx512SortNoMergePathV2SerialTime > 0.0) {
            printfcomma((int)((float)Ct_length/avx512SortNoMergePathV2SerialTime));
        } else if (avx512SortNoMergePathV2SerialTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        // printf("AVX-512 Sort Without Merge Path Branchless:     ");
        // if (avx512SortNoMergePathBranchlessTime > 0.0) {
        //     printfcomma((int)((float)Ct_length/avx512SortNoMergePathBranchlessTime));
        // } else if (avx512SortNoMergePathBranchlessTime == 0.0) {
        //     printf("∞");
        // } else {
        //     printf("N/A");
        // }
        // printf("\n");
        // printf("AVX-512 Sort Without Merge Path Bitonic   :     ");
        // if (avx512SortNoMergePathBitonicTime > 0.0) {
        //     printfcomma((int)((float)Ct_length/avx512SortNoMergePathBitonicTime));
        // } else if (avx512SortNoMergePathBitonicTime == 0.0) {
        //     printf("∞");
        // } else {
        //     printf("N/A");
        // }
        // printf("\n");
        // printf("AVX-512 Sort Without Merge Path AVX       :     ");
        // if (avx512SortNoMergePathavxTime > 0.0) {
        //     printfcomma((int)((float)Ct_length/avx512SortNoMergePathavxTime));
        // } else if (avx512SortNoMergePathavxTime == 0.0) {
        //     printf("∞");
        // } else {
        //     printf("N/A");
        // }
        // printf("\n");
        printf("AVX-512 Merge Sort                      :     ");
        if (avx512MergeSortTime > 0.0) {
            printfcomma((int)((float)Ct_length/avx512MergeSortTime));
        } else if (avx512MergeSortTime == 0.0) {
            printf("∞");
        } else {
            printf("N/A");
        }
        printf("\n");
        #endif
        printf("\n\n");


        printf("Serial Merge           :     %f\n", serialMergeSortTime);
        printf("Serial Merge Branchless:     %f\n", serialMergeNoBranchSortTime);
        printf("Bitonic Merge          :     %f\n", bitonicMergeRealSortTime);
        #ifdef AVX512
        printf("AVX512 Sort W/O MPS    :     %f\n", avx512SortNoMergePathSerialTime);
        printf("AVX512 Sort V2 W/O MPS :     %f\n", avx512SortNoMergePathV2SerialTime);
        // printf("AVX512 Sort W/O MPB    :     %f\n", avx512SortNoMergePathBranchlessTime);
        // printf("AVX512 Sort W/O MPBi   :     %f\n", avx512SortNoMergePathBitonicTime);
        // printf("AVX512 Sort W/O MPA    :     %f\n", avx512SortNoMergePathavxTime);
        printf("AVX512 Merge           :     %f\n", avx512MergeSortTime);
        #endif
    }
}
#endif

#ifdef PARALLELSORT
void parallelTester(
    vec_t** A, uint32_t A_length,
    vec_t** B, uint32_t B_length,
    vec_t** C, uint32_t C_length,
    vec_t** CSorted, uint32_t Ct_length,
    vec_t** CUnsorted, uint32_t runs)
{
    uint32_t numberOfThreads = 0;
    //Check how many threads are running
    #pragma omp parallel
    {
        #pragma omp single
        {
            numberOfThreads = omp_get_num_threads();
        }
    }

    if (!OutToFile) {
        printf("Parameters\n");
        printf("Entropy: %d\n", entropy);
        printf("Runs: %i\n", runs);
        //printf("Array Size: %" PRIu32 "\n", C_length);
        printf("Number of Threads: %i\n", numberOfThreads);
        printf("\n");
    }

    float serialMergeParallelSortTime = 0.0;
    float serialMergeNoBranchParallelSortTime = 0.0;
    float bitonicMergeRealParallelSortTime = 0.0;
    #ifdef AVX512
    float avx512MergeParallelSortNewTime = 0.0;
    float avx512MergeParallelSortTime = 0.0;
    #endif

    float serialMergeParallelSortTimeMax = FLT_MAX;
    float serialMergeNoBranchParallelSortTimeMax = FLT_MAX;
    float bitonicMergeRealParallelSortTimeMax = FLT_MAX;
    #ifdef AVX512
    float avx512MergeParallelSortNewTimeMax = FLT_MAX;
    float avx512MergeParallelSortTimeMax = FLT_MAX;
    #endif

    float temp = 0.0;

    for (uint32_t run = 0; run < RUNS; run++) {
        printf("1\n");
        if (serialMergeParallelSortTime >= 0.0) {
            temp = serialMergeParallelSortTime;
            serialMergeParallelSortTime += testParallelSort<parallelIterativeMergeSort<iterativeMergeSort<serialMerge>,serialMerge>>(
                                                CUnsorted, C_length,
                                                CSorted, Ct_length,
                                                runs, 64, "Standard Parallel Merge Sort");
            temp = serialMergeParallelSortTime - temp;
            if (temp < serialMergeParallelSortTimeMax) {
                serialMergeParallelSortTimeMax = temp;
            }
        }
        printf("2\n");
        if (serialMergeNoBranchParallelSortTime >= 0.0) {
            temp = serialMergeNoBranchParallelSortTime;
            serialMergeNoBranchParallelSortTime += testParallelSort<parallelIterativeMergeSort<iterativeMergeSort<serialMergeNoBranch>,serialMergeNoBranch>>(
                                                        CUnsorted, C_length,
                                                        CSorted, Ct_length,
                                                        runs, 64, "Branchless Merge Sort");
            temp = serialMergeNoBranchParallelSortTime - temp;
            if (temp < serialMergeNoBranchParallelSortTimeMax) {
                serialMergeNoBranchParallelSortTimeMax = temp;
            }
        }
        printf("3\n");
        if (bitonicMergeRealParallelSortTime >= 0.0) {
            temp = bitonicMergeRealParallelSortTime;
            bitonicMergeRealParallelSortTime += testParallelSort<parallelIterativeMergeSort<iterativeMergeSort<bitonicMergeReal>,bitonicMergeReal>>(
                                                    CUnsorted, C_length,
                                                    CSorted, Ct_length,
                                                    runs, 64, "Bitonic Based Merge Sort");
            temp = bitonicMergeRealParallelSortTime - temp;
            if (temp < bitonicMergeRealParallelSortTimeMax) {
                bitonicMergeRealParallelSortTimeMax = temp;
            }
        }
        printf("4\n");
        #ifdef AVX512
        if (avx512MergeParallelSortTime >= 0.0) {
            temp = avx512MergeParallelSortNewTime;
            avx512MergeParallelSortNewTime += testParallelSort<parallelIterativeMergeSort<avx512SortNoMergePathV2<bitonicMergeReal>,bitonicMergeReal>>(
                                                CUnsorted, C_length,
                                                CSorted, Ct_length,
                                                runs, 64, "AVX-512 Based Merge Sort New");
            temp = avx512MergeParallelSortNewTime - temp;
            if (temp < avx512MergeParallelSortNewTimeMax) {
                avx512MergeParallelSortNewTimeMax = temp;
            }
        }
        printf("5\n");
        // if (avx512MergeParallelSortTime >= 0.0) {
        //     temp = avx512MergeParallelSortTime;
        //     avx512MergeParallelSortTime += testParallelSort<parallelIterativeMergeSort<iterativeMergeSort<avx512Merge>,avx512Merge>>(
        //                                         CUnsorted, C_length,
        //                                         CSorted, Ct_length,
        //                                         runs, 64, "AVX-512 Based Merge Sort");
        //     temp = avx512MergeParallelSortTime - temp;
        //     if (temp < avx512MergeParallelSortTimeMax) {
        //         avx512MergeParallelSortTimeMax = temp;
        //     }
        // }
        printf("6\n");
        #endif

        insertData(
            A, A_length,
            B, B_length,
            C, C_length,
            CSorted, Ct_length,
            CUnsorted);
        }

        serialMergeParallelSortTime /= RUNS;
        serialMergeNoBranchParallelSortTime /= RUNS;
        bitonicMergeRealParallelSortTime /= RUNS;
        #ifdef AVX512
        avx512MergeParallelSortNewTime /= RUNS;
        avx512MergeParallelSortTime /= RUNS;
        #endif

        if (OutToFile) {
            writeToParallelSortOut("Standard Merge Sort", entropy, C_length, numberOfThreads, serialMergeParallelSortTime);
            writeToParallelSortOut("Branchless Merge Sort", entropy, C_length, numberOfThreads, serialMergeNoBranchParallelSortTime);
            writeToParallelSortOut("Bitonic Based Merge Sort", entropy, C_length, numberOfThreads, bitonicMergeRealParallelSortTime);
            #ifdef AVX512
            writeToParallelSortOut("AVX-512 Based Merge Sort", entropy, C_length, numberOfThreads, avx512MergeParallelSortTime);
            #endif

            // writeToParallelSortOut("Standard Merge Sort Max", entropy, C_length, numberOfThreads, serialMergeParallelSortTimeMax);
            // writeToParallelSortOut("Branchless Merge Sort Max", entropy, C_length, numberOfThreads, serialMergeNoBranchParallelSortTimeMax);
            // writeToParallelSortOut("Bitonic Based Merge Sort Max", entropy, C_length, numberOfThreads, bitonicMergeRealParallelSortTimeMax);
            #ifdef AVX512
            // writeToParallelSortOut("AVX-512 Based Merge Sort Max", entropy, C_length, numberOfThreads, avx512MergeParallelSortTimeMax);
            #endif
        }

        if (!OutToFile) {
            printf("Parallel Sorting Results:  Elements per Second\n");
            printf("Standard Merge Sort     :     ");
            if (serialMergeParallelSortTime > 0.0) {
                printfcomma((int)((float)Ct_length/serialMergeParallelSortTime));
            } else if (serialMergeParallelSortTime == 0.0) {
                printf("∞");
            } else {
                printf("N/A");
            }
            printf("\n");
            printf("Branchless Merge Sort   :     ");
            if (serialMergeNoBranchParallelSortTime > 0.0) {
                printfcomma((int)((float)Ct_length/serialMergeNoBranchParallelSortTime));
            } else if (serialMergeNoBranchParallelSortTime == 0.0) {
                printf("∞");
            } else {
                printf("N/A");
            }
            printf("\n");
            printf("Bitonic Based Merge Sort:     ");
            if (bitonicMergeRealParallelSortTime > 0.0) {
                printfcomma((int)((float)Ct_length/bitonicMergeRealParallelSortTime));
            } else if (bitonicMergeRealParallelSortTime == 0.0) {
                printf("∞");
            } else {
                printf("N/A");
            }
            printf("\n");
            #ifdef AVX512
            printf("AVX-512 Based Merge Sort:     ");
            if (avx512MergeParallelSortNewTime > 0.0) {
                printfcomma((int)((float)Ct_length/avx512MergeParallelSortNewTime));
            } else if (avx512MergeParallelSortNewTime == 0.0) {
                printf("∞");
            } else {
                printf("N/A");
            }
            printf("\n");
            printf("AVX-512 Based Merge Sort:     ");
            if (avx512MergeParallelSortTime > 0.0) {
                printfcomma((int)((float)Ct_length/avx512MergeParallelSortTime));
            } else if (avx512MergeParallelSortTime == 0.0) {
                printf("∞");
            } else {
                printf("N/A");
            }
            printf("\n");
            #endif
            printf("\n\n");


            printf("Standard Merge Sort Max     :     ");
            if (serialMergeParallelSortTime > 0.0) {
                printfcomma((int)((float)Ct_length/serialMergeParallelSortTimeMax));
            } else if (serialMergeParallelSortTime == 0.0) {
                printf("∞");
            } else {
                printf("N/A");
            }
            printf("\n");
            printf("Branchless Merge Sort Max   :     ");
            if (serialMergeNoBranchParallelSortTime > 0.0) {
                printfcomma((int)((float)Ct_length/serialMergeNoBranchParallelSortTimeMax));
            } else if (serialMergeNoBranchParallelSortTime == 0.0) {
                printf("∞");
            } else {
                printf("N/A");
            }
            printf("\n");
            printf("Bitonic Based Merge Sort Max:     ");
            if (bitonicMergeRealParallelSortTime > 0.0) {
                printfcomma((int)((float)Ct_length/bitonicMergeRealParallelSortTimeMax));
            } else if (bitonicMergeRealParallelSortTime == 0.0) {
                printf("∞");
            } else {
                printf("N/A");
            }
            printf("\n");
            #ifdef AVX512
            printf("AVX-512 Based Merge Sort Max:     ");
            if (avx512MergeParallelSortNewTimeMax > 0.0) {
                printfcomma((int)((float)Ct_length/avx512MergeParallelSortNewTimeMax));
            } else if (avx512MergeParallelSortNewTimeMax == 0.0) {
                printf("∞");
            } else {
                printf("N/A");
            }
            printf("\n");
            printf("AVX-512 Based Merge Sort Max:     ");
            if (avx512MergeParallelSortTime > 0.0) {
                printfcomma((int)((float)Ct_length/avx512MergeParallelSortTimeMax));
            } else if (avx512MergeParallelSortTime == 0.0) {
                printf("∞");
            } else {
                printf("N/A");
            }
            printf("\n");
            #endif
            printf("\n\n");


        }
}
#endif

void MergePathSplitter(
    vec_t * A, uint32_t A_length,
    vec_t * B, uint32_t B_length,
    vec_t * C, uint32_t C_length,
    uint32_t threads, uint32_t* ASplitters, uint32_t* BSplitters)
{
    MergePathSplitter2(A, A_length, B, B_length, C, C_length, threads, ASplitters, BSplitters, 0);
}

void MergePathSplitter2(
    vec_t * A, uint32_t A_length,
    vec_t * B, uint32_t B_length,
    vec_t * C, uint32_t C_length,
    uint32_t threads, uint32_t* ASplitters, uint32_t* BSplitters, uint32_t p)
{
  for (uint32_t i = 0; i <= threads; i++) {
      ASplitters[i] = A_length;
      BSplitters[i] = B_length;
  }

  uint64_t minLength = A_length > B_length ? B_length : A_length;

  for (uint32_t thread=0; thread<threads;thread++)
  {
    uint64_t combinedIndex = thread * (minLength * 2) / threads;
    uint64_t x_top, y_top, x_bottom, current_x, current_y, offset, oldx, oldy;
    x_top = combinedIndex > minLength ? minLength : combinedIndex;
    y_top = combinedIndex > (minLength) ? combinedIndex - (minLength) : 0;
    x_bottom = y_top;

    oldx = -1;
    oldy = -1;

    vec_t Ai, Bi;
    while(1) {
      offset = (x_top - x_bottom) / 2;
      if (x_top < x_bottom) {
          offset = 0;
      }
      current_y = y_top + offset;
      current_x = x_top - offset;

      if (current_x == oldx || current_y == oldy) {
          return;
      }

      oldx = current_x;
      oldy = current_y;

      if(current_x > A_length - 1 || current_y < 1) {
        Ai = 1;Bi = 0;
      } else {
        Ai = A[current_x];Bi = B[current_y - 1];
      }
      if(Ai > Bi) {
        if(current_y > B_length - 1 || current_x < 1) {
          Ai = 0;Bi = 1;
        } else {
          Ai = A[current_x - 1];Bi = B[current_y];
        }

        if(Ai <= Bi) {//Found it
          ASplitters[thread]   = current_x;
          BSplitters[thread] = current_y;
          break;
        } else {//Both zeros
          x_top = current_x - 1;y_top = current_y + 1;
        }
      } else {// Both ones
        x_bottom = current_x + 1;
      }
  }
}

}

#define PRINT_ARRAY_INDEX(ARR,IND) for (int t=0; t<threads;t++){printf("%10d, ",ARR[IND[t]]);}printf("\n");

void hostParseArgs(int argc, char** argv)
{
  static struct option long_options[] = {
    {"Alength", required_argument, 0, 'A'},
    {"Blength", required_argument, 0, 'B'},
    {"Runs"   , optional_argument, 0, 'R'},
    {"Entropy", optional_argument, 0, 'E'},
    {"FileOut", no_argument      , 0, 'F'},
    {"help"   , no_argument      , 0, 'h'},
    {0        , 0                , 0,  0 }
  };

  while(1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "A:B:R:E:h:F",
  long_options, &option_index);
    extern char * optarg;
    //extern int    optind, opterr, optopt;
    int intout = 0;

    if(-1 == c)
      break;

    switch(c) {
      default:
        printf("Unrecognized option: %c\n\n", c);
      case 'h':
        printf("\nCreates two arrays A and B and "
            "merges them into array C in parallel on OpenMP.");
        printf("\n\nUsage"
            "\n====="
            "\n\n\t-A --Alength <number>"
            "\n\t\tSpecify the length of randomly "
            "generated array A.\n"
            "\n\t-B --Blength <number>"
            "\n\t\tSpecify the length of randomly "
            "generated array B.\n"
            "\n\t-R --Runs <number>"
            "\n\t\tSpecify the number of runs to be used for "
            "serial and parallel algorithms.\n"
            "\n\t-E --Entropy <number>"
            "\n\t\tSpecify the number of bits of entropy "
            "to be used for random number generation\n"
            "\n\t-F --FileOut 0 or 1"
            "\n\t\tSpecify weather to write full "
            "output to a file\n"
            );
         exit(0);
         break;
      case 'A':
        errno = 0;
        intout = strtol(optarg, NULL, 10);
        if(errno || intout < 0) {
          printf("Error - Alength %s\n", optarg);
          exit(-1);
        }
        h_ui_A_length = intout;
        break;
      case 'B':
        errno = 0;
        intout = strtol(optarg, NULL, 10);
        if(errno || intout < 0) {
          printf("Error - Blength %s\n", optarg);
          exit(-1);
        }
        h_ui_B_length = intout;
        break;
      case 'R':
        intout = strtol(optarg, NULL, 10);
        if(errno || intout < 0) {
          printf("Error - Runs %s\n", optarg);
          exit(-1);
        }
        RUNS = intout;
        break;
      case 'E':
        intout = strtol(optarg, NULL, 10);
        if(errno || intout < 0) {
          printf("Error - Entropy %s\n", optarg);
          exit(-1);
        }
        entropy = (intout < 1) ? 1: intout;
        break;
      case 'F':
        OutToFile = 1;
        break;
    }
  }
  h_ui_C_length = h_ui_A_length + h_ui_B_length;
  h_ui_Ct_length = h_ui_C_length;
}
