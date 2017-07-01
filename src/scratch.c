static void bitonicAVX512Merge(vec_t* A, uint32_t A_length,
                            vec_t* B, uint32_t B_length,
                            vec_t* C, uint32_t C_length){
	int l, r, p = 0;

    uint32_t* output = C;
    int BSize = (int)B_length
    int ASize = (int)A_length;

	int nbits, leadL, leadR, lead;
	const __m512i vecIndexInc = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
	const __m512i vecReverse = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	const __m512i vecMaxInt = _mm512_set1_epi32(0x7fffffff);
	//const __m512i vecMid = _mm512_set1_epi32(mid);
	//const __m512i vecRight = _mm512_set1_epi32(right);
	const __m512i vecPermuteIndex16 = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13 ,12, 11, 10, 9, 8);
	const __m512i vecPermuteIndex8 = _mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
	const __m512i vecPermuteIndex4 = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
   	const __m512i vecPermuteIndex2 = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
	__m512i vecC, vecD, vecOL, vecOH;
	__m512i vecL1, vecH1, vecL2, vecH2, vecL3, vecH3, vecL4, vecH4;
	__mmask16 vecMaskOL, vecMaskOH;

    /*for short segments*/
    if(ASize < 16 || BSize < 16){
 	      l = 0; r = 0;
  	      while(l < ASize && r < BSize){
      	       if(input[r] < input[l]){
  	                /*save the element from the right segment to temp array*/
    	            output[p++] = input[r++];
     	       }else{
        	        /*save the element from the left segment to temp array*/
        	        output[p++] = input[l++];
      	       }
          }

          /*copy the remaining to the output buffer*/
          if(l < ASize){
              memcpy(output + p, input + l, (ASize - l) * sizeof(uint32_t));
          }else if(r < BSize){
            	memcpy(output + p, input + r, (BSize - r) * sizeof(uint32_t));
          }
	     /*return*/
	  	 return;
	}

	/*use simd vectorization*/
	l = 0; r = 0;
	vecOL = _mm512_load_epi32(A + l);
	vecOH = _mm512_load_epi32(B + r);
	l += 16; r += 16;

	/*enter the core loop*/
	do{
		if(_mm512_reduce_min_epi32(vecOL) >= _mm512_reduce_max_epi32(vecOH)){
			_mm512_store_epi32(output + p, vecOH);
			p += 16;
			vecOH = vecOL;
		}else if(_mm512_reduce_min_epi32(vecOH) >= _mm512_reduce_max_epi32(vecOL)){
			_mm512_store_epi32(output + p, vecOL);
			p += 16;
		}else{
			/*in-register bitonic merge network*/
			vecOH = _mm512_permutevar_epi32(vecReverse, vecOH);	/*reverse B*/

			/*Level 1*/
			vecL1 = _mm512_min_epi32(vecOL, vecOH);
			vecH1 = _mm512_max_epi32(vecOL, vecOH);
			//printVector(vecL1, __LINE__);
			//printVector(vecH1, __LINE__);

			/*Level 2*/
			vecC = _mm512_permutevar_epi32(vecPermuteIndex16, vecL1);
			vecD = _mm512_permutevar_epi32(vecPermuteIndex16, vecH1);
			vecL2 = _mm512_mask_min_epi32(vecL2, 0x00ff, vecC, vecL1);
			vecH2 = _mm512_mask_min_epi32(vecH2, 0x00ff, vecD, vecH1);
			vecL2 = _mm512_mask_max_epi32(vecL2, 0xff00, vecC, vecL1);
			vecH2 = _mm512_mask_max_epi32(vecH2, 0xff00, vecD, vecH1);
			//printVector(vecL2, __LINE__);
			//printVector(vecH2, __LINE__);

			/*Level 3*/
			vecC = _mm512_permutevar_epi32(vecPermuteIndex8, vecL2);
			vecD = _mm512_permutevar_epi32(vecPermuteIndex8, vecH2);
      	    vecL3 = _mm512_mask_min_epi32(vecL3, 0x0f0f, vecC, vecL2);
      	    vecH3 = _mm512_mask_min_epi32(vecH3, 0x0f0f, vecD, vecH2);
      	    vecL3 = _mm512_mask_max_epi32(vecL3, 0xf0f0, vecC, vecL2);
      	    vecH3 = _mm512_mask_max_epi32(vecH3, 0xf0f0, vecD, vecH2);
     		//printVector(vecL3, __LINE__);
     		//printVector(vecH3, __LINE__);

			/*Level 4*/
      	    vecC = _mm512_permutevar_epi32(vecPermuteIndex4, vecL3);
          	vecD = _mm512_permutevar_epi32(vecPermuteIndex4, vecH3);
          	vecL4 = _mm512_mask_min_epi32(vecL4, 0x3333, vecC, vecL3);
          	vecH4 = _mm512_mask_min_epi32(vecH4, 0x3333, vecD, vecH3);
          	vecL4 = _mm512_mask_max_epi32(vecL4, 0xcccc, vecC, vecL3);
          	vecH4 = _mm512_mask_max_epi32(vecH4, 0xcccc, vecD, vecH3);
     		//printVector(vecL4, __LINE__);
     		//printVector(vecH4, __LINE__);

			/*Level 5*/
          	vecC = _mm512_permutevar_epi32(vecPermuteIndex2, vecL4);
          	vecD = _mm512_permutevar_epi32(vecPermuteIndex2, vecH4);
          	vecOL = _mm512_mask_min_epi32(vecOL, 0x5555, vecC, vecL4);
          	vecOH = _mm512_mask_min_epi32(vecOH, 0x5555, vecD, vecH4);
          	vecOL = _mm512_mask_max_epi32(vecOL, 0xaaaa, vecC, vecL4);
          	vecOH = _mm512_mask_max_epi32(vecOH, 0xaaaa, vecD, vecH4);
			//printVector(vecOL, __LINE__);
			//printVector(vecOH, __LINE__);

			/*save vecL to the output vector: always memory aligned*/
			_mm512_store_epi32(output + p, vecOL);
			p += 16;
		}

		/*condition check*/
		if(l + 16 >= ASize || r + 16 >= BSize){
			break;
		}

		/*determine which segment to use*/
		leadL = input[l];
		leadR = input[r];
		lead = _mm512_reduce_max_epi32(vecOH);
		if(lead <= leadL && lead <= leadR){
			_mm512_store_epi32(output + p, vecOH);
			vecOL = _mm512_load_epi32(input + l);
			vecOH = _mm512_load_epi32(input + r);
			p += 16;
			l += 16;
			r += 16;
		}else if(leadR < leadL){
			vecOL = _mm512_load_epi32(input + r);
			r += 16;
		}else{
			vecOL = _mm512_load_epi32(input + l);
			l += 16;
		}
	}while(1);

    while(l < ASize && r < BSize){
        if(input[r] < input[l]){
            output[p++] = input[r++];
        }else{
            output[p++] = input[l++];
        }
    }
	/*copy the remaining to the output buffer*/
	if(l < ASize){
		memcpy(output + p, input + l, (ASize - l) * sizeof(uint32_t));
	}else if(r < BSize){
		memcpy(output + p, input + r, (BSize - r) * sizeof(uint32_t));
	}
}
