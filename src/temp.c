static void mergeOutPlace(uint32_t* input, uint32_t*output, int left, int mid, int right){
		int l, r, p = left;

		int nbits, leadL, leadR, lead;
		const __m512i vecIndexInc = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
		const __m512i vecReverse = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
		const __m512i vecMaxInt = _mm512_set1_epi32(0x7fffffff);
		const __m512i vecMid = _mm512_set1_epi32(mid);
		const __m512i vecRight = _mm512_set1_epi32(right);
		const __m512i vecPermuteIndex16 = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13 ,12, 11, 10, 9, 8);
		const __m512i vecPermuteIndex8 = _mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
	 	const __m512i vecPermuteIndex4 = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
   	    const __m512i vecPermuteIndex2 = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
		__m512i vecA, vecA2, vecB, vecB2, vecC, vecD, vecOL, vecOL2, vecOH, vecOH2;
		__m512i vecL1, vecH1, vecL2, vecH2, vecL3, vecH3, vecL4, vecH4;
		__mmask16 vecMaskA, vecMaskB, vecMaskOL, vecMaskOH;

		/*use simd vectorization*/
		l = left; r = mid;
		vecA = _mm512_load_epi32(input + l); l += 16;
		vecA2 = _mm512_load_epi32(input + l); l += 16;
		vecB = _mm512_load_epi32(input + r); r += 16;
		vecB2 = _mm512_load_epi32(input + r); r += 16;

		/*enter the core loop*/
		do{
			/*prefetch input[l] and input[r]*/
			//#pragma prefetch input

			if(_mm512_reduce_min_epi32(vecA) >= _mm512_reduce_max_epi32(vecB2)){
				_mm512_store_epi32(output + p, vecB); p += 16;
				_mm512_store_epi32(output + p, vecB2); p += 16;
				vecOH = vecA;
				vecOH2 = vecA2;
			}else if(_mm512_reduce_min_epi32(vecB) >= _mm512_reduce_max_epi32(vecA2)){
				_mm512_store_epi32(output + p, vecA); p += 16;
				_mm512_store_epi32(output + p, vecA2); p += 16;
				vecOH = vecB;
				vecOH2 = vecB2;
			}else{
				/*in-register bitonic merge network*/
				vecB = _mm512_permutevar_epi32(vecReverse, vecB);	/*reverse B*/
				vecB2 = _mm512_permutevar_epi32(vecReverse, vecB2);	/*reverse B2*/
				vecC = vecB;
				vecB = vecB2;
				vecB2 = vecC;	/*swap the content of the vectors*/
				//printVector(vecB, __LINE__);
				//printVector(vecB2, __LINE__);

				/*Level 1*/
				vecL1 = _mm512_min_epi32(vecA, vecB);
				vecH1 = _mm512_max_epi32(vecA, vecB);
				vecL2 = _mm512_min_epi32(vecA2, vecB2);
				vecH2 = _mm512_max_epi32(vecA2, vecB2);
				//printVector(vecL1, __LINE__);
				//printVector(vecH1, __LINE__);

				/*Level 2*/
				vecL3 = _mm512_min_epi32(vecL1, vecL2);
				vecL4 = _mm512_max_epi32(vecL1, vecL2);
				vecH3 = _mm512_min_epi32(vecH1, vecH2);
				vecH4 = _mm512_max_epi32(vecH1, vecH2);

				/*Level 3*/
				vecA = _mm512_permutevar_epi32(vecPermuteIndex16, vecL3);
				vecB = _mm512_permutevar_epi32(vecPermuteIndex16, vecL4);
				vecC = _mm512_permutevar_epi32(vecPermuteIndex16, vecH3);
				vecD = _mm512_permutevar_epi32(vecPermuteIndex16, vecH4);
				vecL1 = _mm512_mask_min_epi32(vecL1, 0x00ff, vecA, vecL3);
                vecL2 = _mm512_mask_min_epi32(vecL2, 0x00ff, vecB, vecL4);
                vecH1 = _mm512_mask_min_epi32(vecH1, 0x00ff, vecC, vecH3);
                vecH2 = _mm512_mask_min_epi32(vecH2, 0x00ff, vecD, vecH4);
				vecL1 = _mm512_mask_max_epi32(vecL1, 0xff00, vecA, vecL3);
                vecL2 = _mm512_mask_max_epi32(vecL2, 0xff00, vecB, vecL4);
                vecH1 = _mm512_mask_max_epi32(vecH1, 0xff00, vecC, vecH3);
                vecH2 = _mm512_mask_max_epi32(vecH2, 0xff00, vecD, vecH4);
				//printVector(vecL2, __LINE__);
				//printVector(vecH2, __LINE__);

				/*Level 4*/
				vecA = _mm512_permutevar_epi32(vecPermuteIndex8, vecL1);
				vecB = _mm512_permutevar_epi32(vecPermuteIndex8, vecL2);
                vecC = _mm512_permutevar_epi32(vecPermuteIndex8, vecH1);
                vecD = _mm512_permutevar_epi32(vecPermuteIndex8, vecH2);
              	vecL3 = _mm512_mask_min_epi32(vecL3, 0x0f0f, vecA, vecL1);
				vecL4 = _mm512_mask_min_epi32(vecL4, 0x0f0f, vecB, vecL2);
				vecH3 = _mm512_mask_min_epi32(vecH3, 0x0f0f, vecC, vecH1);
				vecH4 = _mm512_mask_min_epi32(vecH4, 0x0f0f, vecD, vecH2);
                vecL3 = _mm512_mask_max_epi32(vecL3, 0xf0f0, vecA, vecL1);
                vecL4 = _mm512_mask_max_epi32(vecL4, 0xf0f0, vecB, vecL2);
                vecH3 = _mm512_mask_max_epi32(vecH3, 0xf0f0, vecC, vecH1);
                vecH4 = _mm512_mask_max_epi32(vecH4, 0xf0f0, vecD, vecH2);

				/*Level 5*/
                vecA = _mm512_permutevar_epi32(vecPermuteIndex4, vecL3);
                vecB = _mm512_permutevar_epi32(vecPermuteIndex4, vecL4);
              	vecC = _mm512_permutevar_epi32(vecPermuteIndex4, vecH3);
              	vecD = _mm512_permutevar_epi32(vecPermuteIndex4, vecH4);

              	vecL1 = _mm512_mask_min_epi32(vecL1, 0x3333, vecA, vecL3);
              	vecL2 = _mm512_mask_min_epi32(vecL2, 0x3333, vecB, vecL4);
				vecH1 = _mm512_mask_min_epi32(vecH1, 0x3333, vecC, vecH3);
				vecH2 = _mm512_mask_min_epi32(vecH2, 0x3333, vecD, vecH4);
                vecL1 = _mm512_mask_max_epi32(vecL1, 0xcccc, vecA, vecL3);
                vecL2 = _mm512_mask_max_epi32(vecL2, 0xcccc, vecB, vecL4);
                vecH1 = _mm512_mask_max_epi32(vecH1, 0xcccc, vecC, vecH3);
                vecH2 = _mm512_mask_max_epi32(vecH2, 0xcccc, vecD, vecH4);

				/*Level 6*/
                vecA = _mm512_permutevar_epi32(vecPermuteIndex2, vecL1);
                vecB = _mm512_permutevar_epi32(vecPermuteIndex2, vecL2);
              	vecC = _mm512_permutevar_epi32(vecPermuteIndex2, vecH1);
              	vecD = _mm512_permutevar_epi32(vecPermuteIndex2, vecH2);

      	        vecOL = _mm512_mask_min_epi32(vecOL, 0x5555, vecA, vecL1);
				vecOL2 = _mm512_mask_min_epi32(vecOL2, 0x5555, vecB, vecL2);
      	        vecOH = _mm512_mask_min_epi32(vecOH, 0x5555, vecC, vecH1);
				vecOH2 = _mm512_mask_min_epi32(vecOH2, 0x5555, vecD, vecH2);
                vecOL = _mm512_mask_max_epi32(vecOL, 0xaaaa, vecA, vecL1);
                vecOL2 = _mm512_mask_max_epi32(vecOL2, 0xaaaa, vecB, vecL2);
                vecOH = _mm512_mask_max_epi32(vecOH, 0xaaaa, vecC, vecH1);
                vecOH2 = _mm512_mask_max_epi32(vecOH2, 0xaaaa, vecD, vecH2);

				/*save vecL to the output vector: always memory aligned*/
				_mm512_store_epi32(output + p, vecOL); p += 16;
				_mm512_store_epi32(output + p, vecOL2); p += 16;
			}

			/*condition check*/
			if(l + 32 >= mid || r + 32 >= right){
				break;
			}

			/*determine which segment to use*/
			leadL = input[l];
			leadR = input[r];
			lead = _mm512_reduce_max_epi32(vecOH2);
			if(lead < leadL && lead < leadR){
				_mm512_store_epi32(output + p, vecOH); p += 16;
				_mm512_store_epi32(output + p, vecOH2); p += 16;

				vecA = _mm512_load_epi32(input + l); l += 16;
				vecA2 = _mm512_load_epi32(input + l); l += 16;
				vecB = _mm512_load_epi32(input + r); r += 16;
				vecB2 = _mm512_load_epi32(input + r); r += 16;
			}else if(leadR < leadL){
				vecA = vecOH;
				vecA2 = vecOH2;
				vecB = _mm512_load_epi32(input + r); r += 16;
				vecB2 = _mm512_load_epi32(input + r); r += 16;
			}else{
				vecB = vecOH;
				vecB2 = vecOH2;
				vecA = _mm512_load_epi32(input + l); l += 16;
				vecA2 = _mm512_load_epi32(input + l); l += 16;
			}
		}while(1);

		/*use non-vectorized code to process the leftover*/
	    if(l < mid && r < right){
		      if(input[r] < input[l]){
		        /*write vecOH to the left segment*/
		        l -= 16; _mm512_store_epi32(input + l, vecOH2);
						l -= 16; _mm512_store_epi32(input + l, vecOH);
		      }else{
		        /*write vecOH to the right segment*/
		        r -= 16; _mm512_store_epi32(input + r, vecOH2);
						r -= 16; _mm512_store_epi32(input + r, vecOH);
		      }
		    }else if (l < mid){
		      /*write vecOH to the right segment*/
		      r -= 16; _mm512_store_epi32(input + r, vecOH2);
					r -= 16; _mm512_store_epi32(input + r, vecOH);
		    }else if(r < right){
		      /*write vecOH to the left segment*/
		      l -= 16; _mm512_store_epi32(input + l, vecOH2);
					l -= 16; _mm512_store_epi32(input + l, vecOH);
		    }else{
					/*write vecOH to the output as neither segment has leftover*/
					_mm512_store_epi32(output + p, vecOH); p += 16;
					_mm512_store_epi32(output + p, vecOH2);	p += 16;
			}

		/*start serial merge*/
	   	while(l < mid && r < right) {
	    	if(input[r] < input[l]){
	      	/*save the element from the right segment to temp array*/
	   			output[p++] = input[r++];
	    	}else{
	      	/*save the element from the left segment to temp array*/
	     		output[p++] = input[l++];
	   		}
		}

		/*copy the rest to the buffer*/
		if(l < mid){
			memcpy(output + p, input + l, (mid - l) * sizeof(uint32_t));
		}else if(r < right){
			memcpy(output + p, input + r, (right - r) * sizeof(uint32_t));
		}
	}
