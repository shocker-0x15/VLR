#pragma once

#include "common_internal.h"

namespace VLR {
    class SlotManager {
        uint32_t m_numLayers;
        uint32_t m_numLowestFlagBins;
        uint32_t m_numTotalCompiledFlagBins;
        uint32_t* m_flagBins;
        uint32_t* m_offsetsToOR_AND;
        uint32_t* m_numUsedFlagsUnderBinList;
        uint32_t* m_offsetsToNumUsedFlags;
        uint32_t* m_numFlagsInLayerList;

    public:
        SlotManager() :
            m_numLayers(0), m_numLowestFlagBins(0), m_numTotalCompiledFlagBins(0),
            m_flagBins(nullptr), m_offsetsToOR_AND(nullptr),
            m_numUsedFlagsUnderBinList(nullptr), m_offsetsToNumUsedFlags(nullptr),
            m_numFlagsInLayerList(nullptr) {
        }
        ~SlotManager() {
        }

        void initialize(uint32_t numSlots);

        void finalize();

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

        uint32_t getNumUsed() const {
            return m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[m_numLayers - 2]];
        }

        void debugPrint() const;
    };
}
