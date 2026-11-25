// Copyright 2018 The SwiftShader Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "VkPipelineLayout.hpp"

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <vector>

namespace vk {

static std::atomic<uint32_t> layoutIdentifierSerial = { 1 };  // Start at 1. 0 is invalid/void layout.

// Interns a layout's routine-relevant content: one identifier per distinct content,
// for the lifetime of the process. The set of distinct layouts a process creates
// is bounded, so entries are never evicted.
static uint32_t internLayoutContent(std::vector<uint32_t> &&content)
{
	static std::mutex mutex;
	static auto *identifiers = new std::map<std::vector<uint32_t>, uint32_t>();  // Leaked: no exit-time destructor.
	static uint32_t nextIdentifier = 1;  // 0 is invalid/void layout.

	std::lock_guard<std::mutex> lock(mutex);
	auto inserted = identifiers->emplace(std::move(content), nextIdentifier);
	if(inserted.second)
	{
		nextIdentifier++;
	}
	return inserted.first->second;
}

PipelineLayout::PipelineLayout(const VkPipelineLayoutCreateInfo *pCreateInfo, void *mem)
    : identifier(layoutIdentifierSerial++)
    , descriptorSetCount(pCreateInfo->setLayoutCount)
    , pushConstantRangeCount(pCreateInfo->pushConstantRangeCount)
{
	Binding *bindingStorage = reinterpret_cast<Binding *>(mem);
	uint32_t dynamicOffsetIndex = 0;

	descriptorSets[0].bindings = bindingStorage;  // Used in destroy() for deallocation.

	for(uint32_t i = 0; i < pCreateInfo->setLayoutCount; i++)
	{
		if(pCreateInfo->pSetLayouts[i] == VK_NULL_HANDLE)
		{
			continue;
		}
		const vk::DescriptorSetLayout *setLayout = vk::Cast(pCreateInfo->pSetLayouts[i]);
		uint32_t bindingsArraySize = setLayout->getBindingsArraySize();
		descriptorSets[i].bindings = bindingStorage;
		bindingStorage += bindingsArraySize;
		descriptorSets[i].bindingCount = bindingsArraySize;

		for(uint32_t j = 0; j < bindingsArraySize; j++)
		{
			descriptorSets[i].bindings[j].descriptorType = setLayout->getDescriptorType(j);
			descriptorSets[i].bindings[j].offset = setLayout->getBindingOffset(j);
			descriptorSets[i].bindings[j].dynamicOffsetIndex = dynamicOffsetIndex;
			descriptorSets[i].bindings[j].descriptorCount = setLayout->getDescriptorCount(j);

			if(DescriptorSetLayout::IsDescriptorDynamic(descriptorSets[i].bindings[j].descriptorType))
			{
				dynamicOffsetIndex += setLayout->getDescriptorCount(j);
			}
		}
	}

	pushConstantRanges = reinterpret_cast<VkPushConstantRange *>(bindingStorage);
	std::copy(pCreateInfo->pPushConstantRanges, pCreateInfo->pPushConstantRanges + pCreateInfo->pushConstantRangeCount, pushConstantRanges);

	// Everything this layout exposes to routine generation, in a fixed order.
	std::vector<uint32_t> content;
	content.push_back(pCreateInfo->setLayoutCount);
	for(uint32_t i = 0; i < pCreateInfo->setLayoutCount; i++)
	{
		if(pCreateInfo->pSetLayouts[i] == VK_NULL_HANDLE)
		{
			content.push_back(~0u);
			continue;
		}
		content.push_back(descriptorSets[i].bindingCount);
		for(uint32_t j = 0; j < descriptorSets[i].bindingCount; j++)
		{
			const Binding &binding = descriptorSets[i].bindings[j];
			content.push_back(static_cast<uint32_t>(binding.descriptorType));
			content.push_back(binding.offset);
			content.push_back(binding.dynamicOffsetIndex);
			content.push_back(binding.descriptorCount);
		}
	}
	content.push_back(pCreateInfo->pushConstantRangeCount);
	for(uint32_t i = 0; i < pCreateInfo->pushConstantRangeCount; i++)
	{
		content.push_back(pushConstantRanges[i].stageFlags);
		content.push_back(pushConstantRanges[i].offset);
		content.push_back(pushConstantRanges[i].size);
	}
	contentIdentifier = internLayoutContent(std::move(content));

	incRefCount();
}

void PipelineLayout::destroy(const VkAllocationCallbacks *pAllocator)
{
	vk::freeHostMemory(descriptorSets[0].bindings, pAllocator);  // pushConstantRanges are in the same allocation
}

bool PipelineLayout::release(const VkAllocationCallbacks *pAllocator)
{
	if(decRefCount() == 0)
	{
		vk::freeHostMemory(descriptorSets[0].bindings, pAllocator);  // pushConstantRanges are in the same allocation
		return true;
	}
	return false;
}

size_t PipelineLayout::ComputeRequiredAllocationSize(const VkPipelineLayoutCreateInfo *pCreateInfo)
{
	uint32_t bindingsCount = 0;
	for(uint32_t i = 0; i < pCreateInfo->setLayoutCount; i++)
	{
		if(pCreateInfo->pSetLayouts[i] == VK_NULL_HANDLE)
		{
			continue;
		}
		bindingsCount += vk::Cast(pCreateInfo->pSetLayouts[i])->getBindingsArraySize();
	}

	return bindingsCount * sizeof(Binding) +                                   // descriptorSets[]
	       pCreateInfo->pushConstantRangeCount * sizeof(VkPushConstantRange);  // pushConstantRanges[]
}

size_t PipelineLayout::getDescriptorSetCount() const
{
	return descriptorSetCount;
}

uint32_t PipelineLayout::getBindingCount(uint32_t setNumber) const
{
	return descriptorSets[setNumber].bindingCount;
}

uint32_t PipelineLayout::getDynamicOffsetIndex(uint32_t setNumber, uint32_t bindingNumber) const
{
	ASSERT(setNumber < descriptorSetCount && bindingNumber < descriptorSets[setNumber].bindingCount);
	return descriptorSets[setNumber].bindings[bindingNumber].dynamicOffsetIndex;
}

uint32_t PipelineLayout::getDescriptorCount(uint32_t setNumber, uint32_t bindingNumber) const
{
	ASSERT(setNumber < descriptorSetCount && bindingNumber < descriptorSets[setNumber].bindingCount);
	return descriptorSets[setNumber].bindings[bindingNumber].descriptorCount;
}

uint32_t PipelineLayout::getBindingOffset(uint32_t setNumber, uint32_t bindingNumber) const
{
	ASSERT(setNumber < descriptorSetCount && bindingNumber < descriptorSets[setNumber].bindingCount);
	return descriptorSets[setNumber].bindings[bindingNumber].offset;
}

VkDescriptorType PipelineLayout::getDescriptorType(uint32_t setNumber, uint32_t bindingNumber) const
{
	ASSERT(setNumber < descriptorSetCount && bindingNumber < descriptorSets[setNumber].bindingCount);
	return descriptorSets[setNumber].bindings[bindingNumber].descriptorType;
}

uint32_t PipelineLayout::getDescriptorSize(uint32_t setNumber, uint32_t bindingNumber) const
{
	return DescriptorSetLayout::GetDescriptorSize(getDescriptorType(setNumber, bindingNumber));
}

bool PipelineLayout::isDescriptorDynamic(uint32_t setNumber, uint32_t bindingNumber) const
{
	return DescriptorSetLayout::IsDescriptorDynamic(getDescriptorType(setNumber, bindingNumber));
}

uint32_t PipelineLayout::incRefCount()
{
	return ++refCount;
}

uint32_t PipelineLayout::decRefCount()
{
	return --refCount;
}

}  // namespace vk
