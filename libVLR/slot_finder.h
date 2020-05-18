#pragma once

#include "shared/common_internal.h"

namespace VLR {
    class SlotFinder {
        uint32_t m_numLayers;
        uint32_t m_numLowestFlagBins;
        uint32_t m_numTotalCompiledFlagBins;
        uint32_t* m_flagBins;
        uint32_t* m_offsetsToOR_AND;
        uint32_t* m_numUsedFlagsUnderBinList;
        uint32_t* m_offsetsToNumUsedFlags;
        uint32_t* m_numFlagsInLayerList;

        SlotFinder(const SlotFinder &) = delete;
        SlotFinder &operator=(const SlotFinder &) = delete;

        void aggregate();

    public:
        static constexpr uint32_t InvalidSlotIndex = 0xFFFFFFFF;

        SlotFinder() :
            m_numLayers(0), m_numLowestFlagBins(0), m_numTotalCompiledFlagBins(0),
            m_flagBins(nullptr), m_offsetsToOR_AND(nullptr),
            m_numUsedFlagsUnderBinList(nullptr), m_offsetsToNumUsedFlags(nullptr),
            m_numFlagsInLayerList(nullptr) {
        }
        ~SlotFinder() {
        }

        void initialize(uint32_t numSlots);

        void finalize();

        SlotFinder &operator=(SlotFinder &&inst) {
            finalize();

            m_numLayers = inst.m_numLayers;
            m_numLowestFlagBins = inst.m_numLowestFlagBins;
            m_numTotalCompiledFlagBins = inst.m_numTotalCompiledFlagBins;
            m_flagBins = inst.m_flagBins;
            m_offsetsToOR_AND = inst.m_offsetsToOR_AND;
            m_numUsedFlagsUnderBinList = inst.m_numUsedFlagsUnderBinList;
            m_offsetsToNumUsedFlags = inst.m_offsetsToNumUsedFlags;
            m_numFlagsInLayerList = inst.m_numFlagsInLayerList;
            inst.m_flagBins = nullptr;
            inst.m_offsetsToOR_AND = nullptr;
            inst.m_numUsedFlagsUnderBinList = nullptr;
            inst.m_offsetsToNumUsedFlags = nullptr;
            inst.m_numFlagsInLayerList = nullptr;

            return *this;
        }
        SlotFinder(SlotFinder &&inst) {
            *this = std::move(inst);
        }

        void resize(uint32_t numSlots);

        void reset() {
            std::fill_n(m_flagBins, m_numLowestFlagBins + m_numTotalCompiledFlagBins, 0);
            std::fill_n(m_numUsedFlagsUnderBinList, m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2, 0);
        }

        uint32_t getNumLayers() const {
            return m_numLayers;
        }

        const uint32_t* getOffsetsToOR_AND() const {
            return m_offsetsToOR_AND;
        }

        const uint32_t* getOffsetsToNumUsedFlags() const {
            return m_offsetsToNumUsedFlags;
        }

        const uint32_t* getNumFlagsInLayerList() const {
            return m_numFlagsInLayerList;
        }



        void setInUse(uint32_t slotIdx);

        void setNotInUse(uint32_t slotIdx);

        bool getUsage(uint32_t slotIdx) const {
            uint32_t binIdx = slotIdx / 32;
            uint32_t flagIdxInBin = slotIdx % 32;
            uint32_t flagBin = m_flagBins[binIdx];

            return (bool)((flagBin >> flagIdxInBin) & 0x1);
        }

        uint32_t getFirstAvailableSlot() const;

        uint32_t getFirstUsedSlot() const;

        uint32_t find_nthUsedSlot(uint32_t n) const;

        uint32_t getNumSlots() const {
            return m_numFlagsInLayerList[0];
        }

        uint32_t getNumUsed() const {
            return m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[m_numLayers - 1]];
        }

        void debugPrint() const;
    };
}
