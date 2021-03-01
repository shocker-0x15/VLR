#include "slot_finder.h"

namespace vlr {
    void SlotFinder::initialize(uint32_t numSlots) {
        m_numLayers = 1;
        m_numLowestFlagBins = nextMultiplierForPowOf2(numSlots, 32);

        uint32_t numFlagBinsInLayer = m_numLowestFlagBins;
        m_numTotalCompiledFlagBins = 0;
        while (numFlagBinsInLayer > 1) {
            ++m_numLayers;
            numFlagBinsInLayer = nextMultiplierForPowOf2(numFlagBinsInLayer, 32);
            m_numTotalCompiledFlagBins += 2 * numFlagBinsInLayer; // OR bins and AND bins
        }

        size_t memSize = sizeof(uint32_t) * ((m_numLowestFlagBins + m_numTotalCompiledFlagBins) +
                                             (m_numLayers - 1) * 2 +
                                             (m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2) +
                                             (m_numLayers - 1) +
                                             m_numLayers);
        void* mem = malloc(memSize);

        uintptr_t memHead = (uintptr_t)mem;
        m_flagBins = (uint32_t*)memHead;
        memHead += sizeof(uint32_t) * (m_numLowestFlagBins + m_numTotalCompiledFlagBins);

        m_offsetsToOR_AND = (uint32_t*)memHead;
        memHead += sizeof(uint32_t) * (m_numLayers - 1) * 2;

        m_numUsedFlagsUnderBinList = (uint32_t*)memHead;
        memHead += sizeof(uint32_t) * (m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2);

        m_offsetsToNumUsedFlags = (uint32_t*)memHead;
        memHead += sizeof(uint32_t) * (m_numLayers - 1);

        m_numFlagsInLayerList = (uint32_t*)memHead;

        uint32_t layer = 0;
        uint32_t offsetToOR_AND = 0;
        uint32_t offsetToNumUsedFlags = 0;
        {
            m_numFlagsInLayerList[layer] = numSlots;

            numFlagBinsInLayer = nextMultiplierForPowOf2(numSlots, 32);

            offsetToOR_AND += numFlagBinsInLayer;
            offsetToNumUsedFlags += numFlagBinsInLayer;
        }
        while (numFlagBinsInLayer > 1) {
            ++layer;
            m_numFlagsInLayerList[layer] = numFlagBinsInLayer;

            numFlagBinsInLayer = nextMultiplierForPowOf2(numFlagBinsInLayer, 32);

            m_offsetsToOR_AND[2 * (layer - 1) + 0] = offsetToOR_AND;
            m_offsetsToOR_AND[2 * (layer - 1) + 1] = offsetToOR_AND + numFlagBinsInLayer;
            m_offsetsToNumUsedFlags[layer - 1] = offsetToNumUsedFlags;

            offsetToOR_AND += 2 * numFlagBinsInLayer;
            offsetToNumUsedFlags += numFlagBinsInLayer;
        }

        std::fill_n(m_flagBins, m_numLowestFlagBins + m_numTotalCompiledFlagBins, 0);
        std::fill_n(m_numUsedFlagsUnderBinList, m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2, 0);
    }

    void SlotFinder::finalize() {
        free(m_flagBins);
    }

    void SlotFinder::setInUse(uint32_t slotIdx) {
        bool setANDFlag;
        uint32_t flagIdxInLayer = slotIdx;

        // JP: 最下層
        {
            uint32_t binIdx = flagIdxInLayer / 32;
            uint32_t flagIdxInBin = flagIdxInLayer % 32;

            uint32_t &flagBin = m_flagBins[binIdx];
            uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[binIdx];
            VLRAssert(((flagBin >> flagIdxInBin) & 0x1) == 0, "Specified slot is already in use.");
            flagBin |= (1 << flagIdxInBin);
            ++numUsedFlagsUnderBin;

            // JP: このビンに利用可能なスロットが無くなった場合は次のANDレイヤーもフラグを立てる。
            uint32_t numRemainingFlags = m_numFlagsInLayerList[0] - 32 * binIdx;
            uint32_t mask = numRemainingFlags >= 32 ? 0xFFFFFFFF : ((1 << numRemainingFlags) - 1);
            setANDFlag = flagBin == mask;

            flagIdxInLayer = binIdx;
        }

        for (int layer = 1; layer < m_numLayers; ++layer) {
            uint32_t binIdx = flagIdxInLayer / 32;
            uint32_t flagIdxInBin = flagIdxInLayer % 32;

            uint32_t &ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * (layer - 1) + 0] + binIdx];
            uint32_t &ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * (layer - 1) + 1] + binIdx];
            uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[(layer - 1)] + binIdx];
            ORFlagBin |= (1 << flagIdxInBin);
            if (setANDFlag)
                ANDFlagBin |= (1 << flagIdxInBin);
            ++numUsedFlagsUnderBin;

            // JP: このビンに利用可能なスロットが無くなった場合は次のANDレイヤーもフラグを立てる。
            uint32_t numRemainingFlags = m_numFlagsInLayerList[layer] - 32 * binIdx;
            uint32_t mask = numRemainingFlags >= 32 ? 0xFFFFFFFF : ((1 << numRemainingFlags) - 1);
            setANDFlag = (ANDFlagBin & mask) == mask;

            flagIdxInLayer = binIdx;
        }
    }

    void SlotFinder::setNotInUse(uint32_t slotIdx) {
        bool resetORFlag;
        uint32_t flagIdxInLayer = slotIdx;

        // JP: 最下層
        {
            uint32_t binIdx = flagIdxInLayer / 32;
            uint32_t flagIdxInBin = flagIdxInLayer % 32;

            uint32_t &flagBin = m_flagBins[binIdx];
            uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[binIdx];
            VLRAssert(((flagBin >> flagIdxInBin) & 0x1) == 1, "Specified slot is already not in use.");
            flagBin &= ~(1 << flagIdxInBin);
            --numUsedFlagsUnderBin;

            // JP: このビンに使用中スロットが無くなった場合は次のORレイヤーのフラグを下げる。
            uint32_t numRemainingFlags = m_numFlagsInLayerList[0] - 32 * binIdx;
            uint32_t mask = numRemainingFlags >= 32 ? 0xFFFFFFFF : ((1 << numRemainingFlags) - 1);
            resetORFlag = (flagBin & mask) == 0;

            flagIdxInLayer = binIdx;
        }

        for (int layer = 1; layer < m_numLayers; ++layer) {
            uint32_t binIdx = flagIdxInLayer / 32;
            uint32_t flagIdxInBin = flagIdxInLayer % 32;

            uint32_t &ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * (layer - 1) + 0] + binIdx];
            uint32_t &ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * (layer - 1) + 1] + binIdx];
            uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[(layer - 1)] + binIdx];
            if (resetORFlag)
                ORFlagBin &= ~(1 << flagIdxInBin);
            ANDFlagBin &= ~(1 << flagIdxInBin);
            --numUsedFlagsUnderBin;

            // JP: このビンに使用中スロットが無くなった場合は次のORレイヤーのフラグを下げる。
            uint32_t numRemainingFlags = m_numFlagsInLayerList[layer] - 32 * binIdx;
            uint32_t mask = numRemainingFlags >= 32 ? 0xFFFFFFFF : ((1 << numRemainingFlags) - 1);
            resetORFlag = (ORFlagBin & mask) == 0;

            flagIdxInLayer = binIdx;
        }

        bool enableDebugPrint = false;
        if (enableDebugPrint)
            debugPrint();
    }

    uint32_t SlotFinder::getFirstAvailableSlot() const {
        uint32_t startBinIdx = 0;
        for (int layer = m_numLayers - 1; layer > 0; --layer) {
            uint32_t ANDFlagBinOffset = m_offsetsToOR_AND[2 * (layer - 1) + 1];
            uint32_t numRemainingFlags = m_numFlagsInLayerList[layer] - 32 * startBinIdx;
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 32);
            bool found = false;
            for (int binIdx = startBinIdx; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t ANDFlagBin = m_flagBins[ANDFlagBinOffset + binIdx];

                // JP: このビンに利用可能なスロットを発見。
                uint32_t mask = numRemainingFlags >= 32 ? 0xFFFFFFFF : ((1 << numRemainingFlags) - 1);
                if ((ANDFlagBin & mask) != mask) {
                    startBinIdx = _tzcnt_u32(~(ANDFlagBin & mask)) + 32 * binIdx;
                    found = true;
                    break;
                }

                numRemainingFlags -= 32;
            }

            // JP: 利用可能なスロットが見つからなかった。
            if (!found)
                return 0xFFFFFFFF;
        }
        uint32_t slot = 0xFFFFFFFF;
        uint32_t numRemainingFlags = m_numFlagsInLayerList[0] - 32 * startBinIdx;
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[0], 32);
        for (int binIdx = startBinIdx; binIdx < numFlagBinsInLayer; ++binIdx) {
            uint32_t flagBin = m_flagBins[binIdx];

            // JP: このビンに利用可能なスロットを発見。
            uint32_t mask = numRemainingFlags >= 32 ? 0xFFFFFFFF : ((1 << numRemainingFlags) - 1);
            if ((flagBin & mask) != mask) {
                slot = _tzcnt_u32(~(flagBin & mask)) + 32 * binIdx;
                break;
            }

            numRemainingFlags -= 32;
        }

        VLRAssert(slot < m_numFlagsInLayerList[0], "Invalid value.");
        return slot;
    }

    uint32_t SlotFinder::getFirstUsedSlot() const {
        uint32_t startBinIdx = 0;
        for (int layer = m_numLayers - 1; layer > 0; --layer) {
            uint32_t ORFlagBinOffset = m_offsetsToOR_AND[2 * (layer - 1) + 0];
            uint32_t numRemainingFlags = m_numFlagsInLayerList[layer] - 32 * startBinIdx;
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 32);
            bool found = false;
            for (int binIdx = startBinIdx; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t ORFlagBin = m_flagBins[ORFlagBinOffset + binIdx];

                // JP: このビンに使用中のスロットを発見。
                uint32_t mask = numRemainingFlags >= 32 ? 0xFFFFFFFF : ((1 << numRemainingFlags) - 1);
                if ((ORFlagBin & mask) != 0) {
                    startBinIdx = _tzcnt_u32(ORFlagBin & mask) + 32 * binIdx;
                    found = true;
                    break;
                }

                numRemainingFlags -= 32;
            }

            // JP: 使用中スロットが見つからなかった。
            if (!found)
                return 0xFFFFFFFF;
        }
        uint32_t slot = 0xFFFFFFFF;
        uint32_t numRemainingFlags = m_numFlagsInLayerList[0] - 32 * startBinIdx;
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[0], 32);
        for (int binIdx = startBinIdx; binIdx < numFlagBinsInLayer; ++binIdx) {
            uint32_t flagBin = m_flagBins[binIdx];

            // JP: このビンに使用中のスロットを発見。
            uint32_t mask = numRemainingFlags >= 32 ? 0xFFFFFFFF : ((1 << numRemainingFlags) - 1);
            if ((flagBin & mask) != mask) {
                slot = _tzcnt_u32(flagBin & mask) + 32 * binIdx;
                break;
            }

            numRemainingFlags -= 32;
        }

        VLRAssert(slot < m_numFlagsInLayerList[0], "Invalid value.");
        return slot;
    }

    uint32_t SlotFinder::find_nthUsedSlot(uint32_t n) const {
        if (n >= getNumUsed())
            return 0xFFFFFFFF;

        uint32_t startBinIdx = 0;
        uint32_t accNumUsed = 0;
        for (int layer = m_numLayers - 1; layer > 0; --layer) {
            uint32_t numUsedFlagsOffset = m_offsetsToNumUsedFlags[layer - 1];
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 32);
            for (int binIdx = startBinIdx; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[numUsedFlagsOffset + binIdx];

                // JP: 現在のビンの配下にインデックスnの使用中スロットがある。
                if (accNumUsed + numUsedFlagsUnderBin > n) {
                    startBinIdx = 32 * binIdx;
                    break;
                }

                accNumUsed += numUsedFlagsUnderBin;
            }
        }
        uint32_t slot = 0xFFFFFFFF;
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[0], 32);
        for (int binIdx = startBinIdx; binIdx < numFlagBinsInLayer; ++binIdx) {
            uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[binIdx];

            // JP: 現在のビン中にインデックスnの使用中スロットがある。
            if (accNumUsed + numUsedFlagsUnderBin > n) {
                uint32_t flagBin = m_flagBins[binIdx];
                slot = 32 * binIdx + nthSetBit(flagBin, n - accNumUsed);
                break;
            }

            accNumUsed += numUsedFlagsUnderBin;
        }

        VLRAssert(slot < m_numFlagsInLayerList[0], "Invalid value.");
        return slot;
    }

    void SlotFinder::debugPrint() const {
        uint32_t numLowestFlagBins = nextMultiplierForPowOf2(m_numFlagsInLayerList[0], 32);
        vlrprintf("----");
        for (int binIdx = 0; binIdx < numLowestFlagBins; ++binIdx) {
            vlrprintf("------------------------------------");
        }
        vlrprintf("\n");
        for (int layer = m_numLayers - 1; layer > 0; --layer) {
            vlrprintf("layer %u (%u):\n", layer, m_numFlagsInLayerList[layer]);
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 32);
            vlrprintf(" OR:");
            for (int binIdx = 0; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * (layer - 1) + 0] + binIdx];
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
                uint32_t ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * (layer - 1) + 1] + binIdx];
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
                uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[layer - 1] + binIdx];
                vlrprintf("                            %8u", numUsedFlagsUnderBin);
            }
            vlrprintf("\n");
        }
        {
            vlrprintf("layer 0 (%u):\n", m_numFlagsInLayerList[0]);
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[0], 32);
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