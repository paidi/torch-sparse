local SparseTensor = torch.class('torch.SparseTensor')

function SparseTensor:__init(data, offsets)
   self.data = data or torch.Tensor()
   self.offsets = offsets
   self.isSparse = true
end

function SparseTensor:clone()
   return torch.SparseTensor(data:clone(), offsets:clone())
end

function SparseTensor:nDimension()
   if self.offsets:size() > 1 then
      return 2
   else
      return 1
   end
end

function SparseTensor:dim()
   return self:nDimension()
end

function SparseTensor:nElement()
   return self.data:size(2)
end

function SparseTensor:size()
   assert(false, "not implemented")
end

function SparseTensor:copy(other)
   return self.data[{2,{}}]:copy(other.data[{2,{}}])
end

function SparseTensor:fill(value)
   self.data[{2,{}}]:fill(value)
end

function SparseTensor:zero()
   self.data[{2,{}}]:fill(0)
end

function SparseTensor:narrow(dim, index, size)
   assert(dim == 1, 'select only implemented for first dimension')
   assert(index < self.offsets:size(), 'index out of bounds')
   assert(self.offsets:size() > 1,
          'select only implemented for matrices')
   local startIdx = self.offsets[index]
   local endIdx = index+size-1 == self.offsets:size() and self.data:size(1)
   endIdx = endIdx or self.offsets[index+size-1]
   return torch.SparseTensor(self.data:narrow(2,startIdx,endIdx-startIdx+1),
                             self.offsets:narrow(1,index,size))
end

function SparseTensor:select(dim, index)
   assert(dim == 1, 'select only implemented for first dimension')
   assert(self.offsets:size() > 1,
          'select only implemented for matrices')
   assert(index < self.offsets:size(), 'index out of bounds')
   local startIdx = 1+self.offsets[index]
   local endIdx = index == self.offsets:size() and self.data:size(1) or self.offsets[index]
   return torch.SparseTensor(self.data[{{},{startIdx,endIdx}}],
                             torch.Tensor(endIdx-startIdx+1))
end

function SparseTensor:apply(fn)
   self.data[{2,{}}]:apply(fn)
end
