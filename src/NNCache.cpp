/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Michael O and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#include "config.h"

#include <functional>
#include <memory>
#include <vector>

#include "NNCache.h"
#include "Utils.h"
#include "UCTSearch.h"
#include "GTP.h"
#include "zlib.h"

const int NNCache::MAX_CACHE_COUNT;
const int NNCache::MIN_CACHE_COUNT;
const size_t NNCache::ENTRY_SIZE;

NNCache::NNCache(int size) : m_size(size) {}

bool NNCache::lookup(std::uint64_t hash, Netresult & result) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ++m_lookups;

    auto iter = m_cache.find(hash);
    if (iter == m_cache.end()) {
        return false;  // Not found.
    }

    const auto& entry = iter->second;

    // Found it.
    ++m_hits;
    result = entry->result;
    return true;
}

void NNCache::insert(std::uint64_t hash,
                     const Netresult& result) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_cache.find(hash) != m_cache.end()) {
        return;  // Already in the cache.
    }

    m_cache.emplace(hash, std::make_unique<Entry>(result));
    m_order.push_back(hash);
    ++m_inserts;

    // If the cache is too large, remove the oldest entry.
    if (m_order.size() > m_size) {
        m_cache.erase(m_order.front());
        m_order.pop_front();
    }
}

void NNCache::resize(int size) {
    m_size = size;
    while (m_order.size() > m_size) {
        m_cache.erase(m_order.front());
        m_order.pop_front();
    }
}

void NNCache::set_size_from_playouts(int max_playouts) {
    // cache hits are generally from last several moves so setting cache
    // size based on playouts increases the hit rate while balancing memory
    // usage for low playout instances. 150'000 cache entries is ~208 MiB
    constexpr auto num_cache_moves = 3;
    auto max_playouts_per_move =
        std::min(max_playouts,
                 UCTSearch::UNLIMITED_PLAYOUTS / num_cache_moves);
    auto max_size = num_cache_moves * max_playouts_per_move;
    max_size = std::min(MAX_CACHE_COUNT, std::max(MIN_CACHE_COUNT, max_size));
    resize(max_size);
}

void NNCache::dump_stats() {
    Utils::myprintf(
        "NNCache: %d/%d hits/lookups = %.1f%% hitrate, %d inserts, %u size\n",
        m_hits, m_lookups, 100. * m_hits / (m_lookups + 1),
        m_inserts, m_cache.size());
}

size_t NNCache::get_estimated_size() {
    return m_order.size() * NNCache::ENTRY_SIZE;
}

void NNCache::save_cache(const std::string& filename) {
    const auto out_str = get_cache();
    const auto buffer_size = out_str.size();
    auto buffer = std::vector<char>(buffer_size);
    memcpy(buffer.data(), out_str.data(), buffer_size);

    auto out = gzopen(filename.c_str(), "wb9");
    Utils::myprintf("Compressing cache...\n");
    const auto comp_size = gzwrite(out, buffer.data(), buffer_size);
    if (!comp_size) {
        throw std::runtime_error("Error in gzip output");
    }
    gzclose(out);
    Utils::myprintf("Saved cache file %s\n", filename.c_str());
}

void NNCache::load_cache(const std::string& filename) {
    // gzopen supports both gz and non-gz files, will decompress
    // or just read directly as needed.
    const auto in = gzopen(filename.c_str(), "rb");
    if (in == nullptr) {
        Utils::myprintf("Could not open cache file: %s\n", filename.c_str());
        return;
    }
    auto in_str = std::stringstream{};
    constexpr auto buffer_size = 64 * 1024;
    auto buffer = std::vector<char>(buffer_size);
    Utils::myprintf("Decompressing cache...\n");
    while (true) {
        const auto bytes_read = gzread(in, buffer.data(), buffer_size);
        if (bytes_read == 0) break;
        if (bytes_read < 0) {
            Utils::myprintf("Failed to decompress or read cache file\n");
            gzclose(in);
            return;
        }
        assert(bytes_read <= buffer_size);
        in_str.write(buffer.data(), bytes_read);
    }
    gzclose(in);

    set_cache(in_str);
    Utils::myprintf("Loaded cache file %s\n", filename.c_str());
}

std::string NNCache::get_cache(void) {
    auto out_str = std::stringstream{};
    out_str << m_size << ' ';

    for (const auto& hash : m_order) {
        auto result = Network::Netresult{};
        lookup(hash, result);
        out_str << hash << ' ';
        for (const auto& policy_move : result.policy) {
            out_str << policy_move << ' ';
        }
        out_str << result.policy_pass << ' ';
        out_str << result.winrate << ' ';
    }
    return out_str.str();
}
void NNCache::set_cache(std::stringstream& in_str) {
    auto size = size_t{};
    in_str >> size;
    if (in_str.fail()) {
        // Empty file?
        return;
    }
    resize(size);

    for (auto i = 0; i < size; ++i) {
        auto hash = std::uint64_t{};
        auto result = Network::Netresult{};
        in_str >> hash;
        for (auto& policy_move : result.policy) {
            in_str >> policy_move;
        }
        in_str >> result.policy_pass;
        in_str >> result.winrate;
        if (in_str.fail()) {
            // Entries don't fill the cache
            return;
        }
        insert(hash, result);
    }
}
