#include "slot_finder.h"

namespace VLR {
    void SlotFinder::initialize(uint32_t numSlots) {
        m_numLayers = 1;
        m_numLowestFlagBins = nextMultiplierForPowOf2(numSlots, 5);

        // e.g. factor 4
        // 0 | 1101 | 0011 | 1001 | 1011 | 0010 | 1010 | 0000 | 1011 | 1110 | 0101 | 111* | **** | **** | **** | **** | **** | 43 flags
        // OR bins:
        // 1 | 1      1      1      1    | 1      1      0      1    | 1      1      1      *    | *      *      *      *    | 11
        // 2 | 1                           1                           1                           *                         | 3
        // AND bins
        // 1 | 0      0      0      0    | 0      0      0      0    | 0      0      1      *    | *      *      *      *    | 11
        // 2 | 0                           0                           0                           *                         | 3
        //
        // numSlots: 43
        // numLowestFlagBins: 11
        // numLayers: 3
        //
        // Memory Order
        // LowestFlagBins (layer 0) | OR, AND Bins (layer 1) | ... | OR, AND Bins (layer n-1)
        // Offset Pair to OR, AND (layer 0) | ... | Offset Pair to OR, AND (layer n-1)
        // NumUsedFlags (layer 0) | ... | NumUsedFlags (layer n-1)
        // Offset to NumUsedFlags (layer 0) | ... | Offset to NumUsedFlags (layer n-1)
        // NumFlags (layer 0) | ... | NumFlags (layer n-1)

        uint32_t numFlagBinsInLayer = m_numLowestFlagBins;
        m_numTotalCompiledFlagBins = 0;
        while (numFlagBinsInLayer > 1) {
            ++m_numLayers;
            numFlagBinsInLayer = nextMultiplierForPowOf2(numFlagBinsInLayer, 5);
            m_numTotalCompiledFlagBins += 2 * numFlagBinsInLayer; // OR bins and AND bins
        }

        size_t memSize = sizeof(uint32_t) *
            ((m_numLowestFlagBins + m_numTotalCompiledFlagBins) +
             m_numLayers * 2 +
             (m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2) +
             m_numLayers +
             m_numLayers);
        void* mem = malloc(memSize);

        uintptr_t memHead = (uintptr_t)mem;
        m_flagBins = (uint32_t*)memHead;
        memHead += sizeof(uint32_t) * (m_numLowestFlagBins + m_numTotalCompiledFlagBins);

        m_offsetsToOR_AND = (uint32_t*)memHead;
        memHead += sizeof(uint32_t) * m_numLayers * 2;

        m_numUsedFlagsUnderBinList = (uint32_t*)memHead;
        memHead += sizeof(uint32_t) * (m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2);

        m_offsetsToNumUsedFlags = (uint32_t*)memHead;
        memHead += sizeof(uint32_t) * m_numLayers;

        m_numFlagsInLayerList = (uint32_t*)memHead;

        uint32_t layer = 0;
        uint32_t offsetToOR_AND = 0;
        uint32_t offsetToNumUsedFlags = 0;
        {
            m_numFlagsInLayerList[layer] = numSlots;

            numFlagBinsInLayer = nextMultiplierForPowOf2(numSlots, 5);

            m_offsetsToOR_AND[2 * layer + 0] = offsetToOR_AND;
            m_offsetsToOR_AND[2 * layer + 1] = offsetToOR_AND;
            m_offsetsToNumUsedFlags[layer] = offsetToNumUsedFlags;

            offsetToOR_AND += numFlagBinsInLayer;
            offsetToNumUsedFlags += numFlagBinsInLayer;
        }
        while (numFlagBinsInLayer > 1) {
            ++layer;
            m_numFlagsInLayerList[layer] = numFlagBinsInLayer;

            numFlagBinsInLayer = nextMultiplierForPowOf2(numFlagBinsInLayer, 5);

            m_offsetsToOR_AND[2 * layer + 0] = offsetToOR_AND;
            m_offsetsToOR_AND[2 * layer + 1] = offsetToOR_AND + numFlagBinsInLayer;
            m_offsetsToNumUsedFlags[layer] = offsetToNumUsedFlags;

            offsetToOR_AND += 2 * numFlagBinsInLayer;
            offsetToNumUsedFlags += numFlagBinsInLayer;
        }

        std::fill_n(m_flagBins, m_numLowestFlagBins + m_numTotalCompiledFlagBins, 0);
        std::fill_n(m_numUsedFlagsUnderBinList, m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2, 0);
    }

    void SlotFinder::finalize() {
        if (m_flagBins)
            free(m_flagBins);
        m_flagBins = nullptr;
    }

    void SlotFinder::aggregate() {
        uint32_t offsetToOR_last = m_offsetsToOR_AND[2 * 0 + 0];
        uint32_t offsetToAND_last = m_offsetsToOR_AND[2 * 0 + 1];
        uint32_t offsetToNumUsedFlags_last = m_offsetsToNumUsedFlags[0];
        for (int layer = 1; layer < m_numLayers; ++layer) {
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
            uint32_t offsetToOR = m_offsetsToOR_AND[2 * layer + 0];
            uint32_t offsetToAND = m_offsetsToOR_AND[2 * layer + 1];
            uint32_t offsetToNumUsedFlags = m_offsetsToNumUsedFlags[layer];
            for (int binIdx = 0; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t &ORFlagBin = m_flagBins[offsetToOR + binIdx];
                uint32_t &ANDFlagBin = m_flagBins[offsetToAND + binIdx];
                uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[offsetToNumUsedFlags + binIdx];

                uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
                for (int bit = 0; bit < numFlagsInBin; ++bit) {
                    uint32_t lBinIdx = 32 * binIdx + bit;
                    uint32_t lORFlagBin = m_flagBins[offsetToOR_last + lBinIdx];
                    uint32_t lANDFlagBin = m_flagBins[offsetToAND_last + lBinIdx];
                    uint32_t lNumFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer - 1] - 32 * lBinIdx);
                    if (lORFlagBin != 0)
                        ORFlagBin |= 1 << bit;
                    if (popcnt(lANDFlagBin) == lNumFlagsInBin)
                        ANDFlagBin |= 1 << bit;
                    numUsedFlagsUnderBin += m_numUsedFlagsUnderBinList[offsetToNumUsedFlags_last + lBinIdx];
                }
            }

            offsetToOR_last = offsetToOR;
            offsetToAND_last = offsetToAND;
            offsetToNumUsedFlags_last = offsetToNumUsedFlags;
        }
    }

    void SlotFinder::resize(uint32_t numSlots) {
        if (numSlots == m_numFlagsInLayerList[0])
            return;

        SlotFinder newFinder;
        newFinder.initialize(numSlots);

        uint32_t numLowestFlagBins = std::min(m_numLowestFlagBins, newFinder.m_numLowestFlagBins);
        for (int binIdx = 0; binIdx < numLowestFlagBins; ++binIdx) {
            uint32_t numFlagsInBin = std::min(32u, numSlots - 32 * binIdx);
            uint32_t mask = numFlagsInBin >= 32 ? 0xFFFFFFFF : ((1 << numFlagsInBin) - 1);
            uint32_t value = m_flagBins[0 + binIdx] & mask;
            newFinder.m_flagBins[0 + binIdx] = value;
            newFinder.m_numUsedFlagsUnderBinList[0 + binIdx] = popcnt(value);
        }

        newFinder.aggregate();

        *this = std::move(newFinder);
    }

    void SlotFinder::setInUse(uint32_t slotIdx) {
        if (getUsage(slotIdx))
            return;

        bool setANDFlag = false;
        uint32_t flagIdxInLayer = slotIdx;
        for (int layer = 0; layer < m_numLayers; ++layer) {
            uint32_t binIdx = flagIdxInLayer / 32;
            uint32_t flagIdxInBin = flagIdxInLayer % 32;

            // JP: 最下層ではOR/ANDは同じ実体だがsetANDFlagが初期値falseであるので設定は1回きり。
            uint32_t &ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 0] + binIdx];
            uint32_t &ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 1] + binIdx];
            uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[layer] + binIdx];
            ORFlagBin |= (1 << flagIdxInBin);
            if (setANDFlag)
                ANDFlagBin |= (1 << flagIdxInBin);
            ++numUsedFlagsUnderBin;

            // JP: このビンに利用可能なスロットが無くなった場合は次のANDレイヤーもフラグを立てる。
            uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
            setANDFlag = popcnt(ANDFlagBin) == numFlagsInBin;

            flagIdxInLayer = binIdx;
        }
    }

    void SlotFinder::setNotInUse(uint32_t slotIdx) {
        if (!getUsage(slotIdx))
            return;

        bool resetORFlag = false;
        uint32_t flagIdxInLayer = slotIdx;
        for (int layer = 0; layer < m_numLayers; ++layer) {
            uint32_t binIdx = flagIdxInLayer / 32;
            uint32_t flagIdxInBin = flagIdxInLayer % 32;

            // JP: 最下層ではOR/ANDは同じ実体だがresetORFlagが初期値falseであるので設定は1回きり。
            uint32_t &ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 0] + binIdx];
            uint32_t &ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 1] + binIdx];
            uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[layer] + binIdx];
            if (resetORFlag)
                ORFlagBin &= ~(1 << flagIdxInBin);
            ANDFlagBin &= ~(1 << flagIdxInBin);
            --numUsedFlagsUnderBin;

            // JP: このビンに使用中スロットが無くなった場合は次のORレイヤーのフラグを下げる。
            uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
            resetORFlag = ORFlagBin == 0;

            flagIdxInLayer = binIdx;
        }
    }

    uint32_t SlotFinder::getFirstAvailableSlot() const {
        uint32_t binIdx = 0;
        for (int layer = m_numLayers - 1; layer >= 0; --layer) {
            uint32_t ANDFlagBinOffset = m_offsetsToOR_AND[2 * layer + 1];
            uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
            uint32_t ANDFlagBin = m_flagBins[ANDFlagBinOffset + binIdx];

            if (popcnt(ANDFlagBin) != numFlagsInBin) {
                // JP: このビンに利用可能なスロットを発見。
                binIdx = tzcnt(~ANDFlagBin) + 32 * binIdx;
            }
            else {
                // JP: 利用可能なスロットが見つからなかった。
                return 0xFFFFFFFF;
            }
        }

        VLRAssert(binIdx < m_numFlagsInLayerList[0], "Invalid value.");
        return binIdx;
    }

    uint32_t SlotFinder::getFirstUsedSlot() const {
        uint32_t binIdx = 0;
        for (int layer = m_numLayers - 1; layer >= 0; --layer) {
            uint32_t ORFlagBinOffset = m_offsetsToOR_AND[2 * layer + 0];
            uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
            uint32_t ORFlagBin = m_flagBins[ORFlagBinOffset + binIdx];

            if (ORFlagBin != 0) {
                // JP: このビンに使用中のスロットを発見。
                binIdx = tzcnt(ORFlagBin) + 32 * binIdx;
            }
            else {
                // JP: 使用中スロットが見つからなかった。
                return 0xFFFFFFFF;
            }
        }

        VLRAssert(binIdx < m_numFlagsInLayerList[0], "Invalid value.");
        return binIdx;
    }

    uint32_t SlotFinder::find_nthUsedSlot(uint32_t n) const {
        if (n >= getNumUsed())
            return 0xFFFFFFFF;

        uint32_t startBinIdx = 0;
        uint32_t accNumUsed = 0;
        for (int layer = m_numLayers - 1; layer >= 0; --layer) {
            uint32_t numUsedFlagsOffset = m_offsetsToNumUsedFlags[layer];
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
            for (int binIdx = startBinIdx; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[numUsedFlagsOffset + binIdx];

                // JP: 現在のビンの配下にインデックスnの使用中スロットがある。
                if (accNumUsed + numUsedFlagsUnderBin > n) {
                    startBinIdx = 32 * binIdx;
                    if (layer == 0) {
                        uint32_t flagBin = m_flagBins[binIdx];
                        startBinIdx += nthSetBit(flagBin, n - accNumUsed);
                    }
                    break;
                }

                accNumUsed += numUsedFlagsUnderBin;
            }
        }

        VLRAssert(startBinIdx < m_numFlagsInLayerList[0], "Invalid value.");
        return startBinIdx;
    }

    void SlotFinder::debugPrint() const {
        uint32_t numLowestFlagBins = nextMultiplierForPowOf2(m_numFlagsInLayerList[0], 5);
        vlrprintf("----");
        for (int binIdx = 0; binIdx < numLowestFlagBins; ++binIdx) {
            vlrprintf("------------------------------------");
        }
        vlrprintf("\n");
        for (int layer = m_numLayers - 1; layer > 0; --layer) {
            vlrprintf("layer %u (%u):\n", layer, m_numFlagsInLayerList[layer]);
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
            vlrprintf(" OR:");
            for (int binIdx = 0; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 0] + binIdx];
                for (int i = 0; i < 32; ++i) {
                    if (i % 8 == 0)
                        vlrprintf(" ");

                    bool valid = binIdx * 32 + i < m_numFlagsInLayerList[layer];
                    if (!valid)
                        continue;

                    bool b = (ORFlagBin >> i) & 0x1;
                    vlrprintf("%c", b ? '|' : '_');
                }
            }
            vlrprintf("\n");
            vlrprintf("AND:");
            for (int binIdx = 0; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 1] + binIdx];
                for (int i = 0; i < 32; ++i) {
                    if (i % 8 == 0)
                        vlrprintf(" ");

                    bool valid = binIdx * 32 + i < m_numFlagsInLayerList[layer];
                    if (!valid)
                        continue;

                    bool b = (ANDFlagBin >> i) & 0x1;
                    vlrprintf("%c", b ? '|' : '_');
                }
            }
            vlrprintf("\n");
            vlrprintf("    ");
            for (int binIdx = 0; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[layer] + binIdx];
                vlrprintf("                            %8u", numUsedFlagsUnderBin);
            }
            vlrprintf("\n");
        }
        {
            vlrprintf("layer 0 (%u):\n", m_numFlagsInLayerList[0]);
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[0], 5);
            vlrprintf("   :");
            for (int binIdx = 0; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t ORFlagBin = m_flagBins[binIdx];
                for (int i = 0; i < 32; ++i) {
                    if (i % 8 == 0)
                        vlrprintf(" ");

                    bool valid = binIdx * 32 + i < m_numFlagsInLayerList[0];
                    if (!valid)
                        continue;

                    bool b = (ORFlagBin >> i) & 0x1;
                    vlrprintf("%c", b ? '|' : '_');
                }
            }
            vlrprintf("\n");
            vlrprintf("    ");
            for (int binIdx = 0; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[binIdx];
                vlrprintf("                            %8u", numUsedFlagsUnderBin);
            }
            vlrprintf("\n");
        }
    }
}