SetBoundSSE proto n:dword, x:ptr real4, bounds:dword

.data
	onef	real4 1.0f
	fourf	real4 4.0f

.code

	AddSourceAsm proc frame n:dword, x:qword, x0:qword, delta:real4 
		sub rsp, 28h			; adust stack pointer for shadow stack space
	.allocstack 28h
	.endprolog
		add rcx, 2				; add 2 to n
		mov rax, rcx			; 
		imul rax, rax			; (n + 2) * (n + 2)
		shufps xmm3, xmm3, 0	; broadcast delta to all elements
		xor r9, r9				; zero out r9 for use as count register
		loopbegin: 
			movaps xmm0, xmmword ptr [r8 + 4 * r9]
			mulps xmm0, xmm3
			addps xmm0, xmmword ptr [rdx + 4 * r9]
			movaps xmmword ptr [rdx + 4 * r9], xmm0
			add r9, 4
			cmp r9, rax
			jl loopbegin
		add rsp, 28h
		ret 0
	AddSourceAsm endp

	RelaxAsm proc frame n:dword, x:ptr real4, x0:ptr real4, a:real4, b:real4, bounds:dword
		push r12
	.pushreg r12
		push r13
	.pushreg r13
		push r14
	.pushreg r14
		sub rsp, 60h
	.allocstack 60h
	.endprolog

		mov rax, rcx						; width = n + 2
		add rax, 2
		mov r12, 20
		loopk:
			movss xmm0, fourf
			mulss xmm0, xmm3					; 4.0f * a
			movss xmm1, dword ptr [rsp + 0A0h]	; b
			addss xmm0, xmm1
			movss xmm1, onef
			divss xmm1, xmm0
			shufps xmm1, xmm1, 0				; broadcast 1.0f / (b + 4.0f * a)
			shufps xmm3, xmm3, 0				; broadcast a
			mov r13, 1
			loopj:
				mov r14, 1
				loopi:
					mov r9, r13
					imul r9, rax
					add r9, r14
					movups xmm0, xmmword ptr [rdx + r9 * 4 - 4]
					movups xmm2, xmmword ptr [rdx + r9 * 4 + 4]
					addps xmm0, xmm2
					sub r9, rax
					movups xmm2, xmmword ptr [rdx + r9 * 4]
					addps xmm0, xmm2
					add r9, rax
					add r9, rax
					movups xmm2, xmmword ptr [rdx + r9 * 4]
					addps xmm0, xmm2
					mulps xmm0, xmm3									; multiply by a
					sub r9, rax
					movups xmm2, xmmword ptr [r8 + r9 * 4]				; x0[i, j]
					addps xmm0, xmm2
					mulps xmm0, xmm1
					movups [rdx + r9 * 4], xmm0
					add r14, 4
					cmp r14, rcx
					jle loopi
				inc r13
				cmp r13, rcx
				jle loopj
			mov [rsp + 28h], rax
			mov [rsp + 30h], rcx
			mov [rsp + 38h], rdx
			mov [rsp + 40h], r8
			mov [rsp + 48h], r9
			mov r8d, dword ptr [rsp + 0A8h]			; bounds
			call SetBoundSSE
			mov rax, [rsp + 28h]
			mov rcx, [rsp + 30h]
			mov rdx, [rsp + 38h]
			mov r8,  [rsp + 40h]
			mov r9,  [rsp + 48h]
			dec r12
			cmp r12, 0
			jne loopk

		add rsp, 60h
		pop r14
		pop r13
		pop r12
		ret 0
	RelaxAsm endp

end