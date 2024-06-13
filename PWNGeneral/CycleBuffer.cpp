// (c) Pawlin Technologies Ltd. 2010
// http://www.pawlin.ru
// File: CycleBuffer.h, c.pp 
// Purpose: Header file for CycleBuffer class, that implements cicular Buffer statistics
// Author: Pavel Skribtsov
// ALL RIGHTS RESERVED. USAGE OF THIS FILE FOR ANY REASON
// WITHOUT WRITTEN PERMISSION FROM
// PAWLIN TECHNOLOGIES LTD IS PROHIBITED
// ANY VIOLATION OF THE COPYRIGHT AND PATENT LAW WILL BE PROSECUTED
// FOR ADDITIONAL INFORMATION REFER www.pawlin.ru

#include "stdafx.h"
#include <vector>
#include "CycleBuffer.h"

//template< typename T >
//T CycleBufferGeneric<T>::getAverage() const {
//	if(fillmode) {
//		if(head_index==0) return 0;
//		return sum / head_index;
//	}
//	return sum / size;
//}
//
//template< typename T >
//void CycleBufferGeneric<T>::push(T value) {
//	sum += value;
//	if(fillmode) {
//		Buffer[head_index] = value;
//		if(head_index==size-1) fillmode = false; // made full cycle
//	}
//	else {
//		T tail_value = Buffer[head_index];
//		Buffer[head_index] = value;
//		sum -= tail_value;
//	}
//	head_index = (head_index + 1) % size;
//}
//
//template< typename T >
//bool CycleBufferGeneric<T>::tryGetTail(T& tailOut) const
//{
//	float tail_value(0);
//
//	if(!fillmode)
//		tail_value = Buffer[head_index];
//
//	tailOut = tail_value;
//
//	return !fillmode;
//}