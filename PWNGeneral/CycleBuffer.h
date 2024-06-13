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
#pragma once
#include <PWNGeneral/PWNExceptionBasic.h>
#include <stddef.h>
#include <vector>

template< typename T >
class CycleBufferGeneric{
	protected:
	size_t size;
	T *Buffer;	
	size_t head_index;
	T sum;
	
	bool fillmode;
	void updateIndex(); 
public:
	CycleBufferGeneric(size_t size) : size(size), Buffer(new T[size]), sum(T()), fillmode(true),head_index(0) {}
	CycleBufferGeneric(size_t size, T* newBufferToOwn, const T& zero_value) : 
		size(size), 
		Buffer(newBufferToOwn), 
		sum(zero_value), 
		fillmode(true),
		head_index(0) 
	{}
	void reset() { sum = 0;  fillmode = true; head_index = 0;} 
	bool empty() const { return fillmode && head_index==0; }
	bool full() const { return !fillmode; }
	virtual ~CycleBufferGeneric() {delete []Buffer;}
	size_t getHeadIndex() const { return head_index; }
	size_t getFilledSize() const { if(fillmode) return head_index; return size; }
	const T& getDelayedValue(size_t delay) const {
		if (delay >= this->cur_size()) throw std::runtime_error("getDelayedValue out of bounds");
		size_t index = (head_index + size - 1 -delay) % size;
		return Buffer[index];
	}
	void updateLastValue(const T &value) {
		if(empty()) push(value);
		else {
			size_t last_head_index = (head_index + size - 1) % size;
			sum += value - Buffer[last_head_index];
			Buffer[last_head_index] = value;
		}
	}
	void push(const T &value) 
	{
		sum += value;
		if(fillmode) {
			Buffer[head_index] = value;
			if(head_index==size-1) fillmode = false; // made full cycle
		}
		else {
			T tail_value = Buffer[head_index];
			Buffer[head_index] = value;
			sum -= tail_value;
		}
		head_index = (head_index + 1) % size;
	}
	size_t max_size() const { return size; }
	T getSum() const { return sum; }
	size_t cur_size() const {
		if (fillmode) return head_index;
		return size;
	}
	T getAverage() const 
	{
		if(fillmode) {
			if(head_index==0) throw( PWNExceptionBasic("CycleBuffer::getAverage - access when empty" )); // return T(0);
			return sum / head_index;
		}
		return sum / size;
	}

	void getAverage(T &out) const 
	{
		out = sum;
		if(fillmode) {
			if(head_index==0) throw( PWNExceptionBasic("CycleBuffer::getAverage - access when empty" )); // return T(0);
			out /= head_index;
			return;
		}
		out /= size;
		return;
	}

	bool tryGetTail(T& tailOut) const
	{
		//T tail_value(0);

		if(fillmode) return false;
		
		tailOut = Buffer[head_index];

		return true;
	}

	bool getWholeBuffer(std::vector<T> &out) const
	{
		if(fillmode) return false;
		out.resize(size);
		T* outPtr = &out.front();
		memcpy(outPtr, Buffer + head_index, sizeof(T) * (size - head_index));
		if(head_index != 0)
			memcpy(outPtr + (size - head_index), Buffer , sizeof(T) * (head_index));
		return true;
	}
};


typedef CycleBufferGeneric<float> CycleBuffer;