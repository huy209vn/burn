//! LRU cache implementation for block management

use alloc::collections::BTreeMap;

/// Least Recently Used (LRU) cache
///
/// Generic cache with O(log n) operations using BTreeMap (no_std compatible).
/// Tracks access times to evict least recently used items when capacity is reached.
///
/// # Example
///
/// ```ignore
/// let mut cache = LRUCache::new(100);
///
/// cache.insert(key, value);
///
/// if let Some(val) = cache.get(&key) {
///     // Use cached value
/// }
/// ```
#[derive(Debug, Clone)]
pub struct LRUCache<K, V> {
    /// Maximum capacity (number of items)
    capacity: usize,

    /// Cache storage: key → (value, access_time)
    cache: BTreeMap<K, CacheEntry<V>>,

    /// Global access counter (monotonically increasing)
    access_counter: u64,
}

/// Cache entry with value and access metadata
#[derive(Debug, Clone)]
struct CacheEntry<V> {
    value: V,
    access_time: u64,
}

impl<K, V> LRUCache<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    /// Create new LRU cache with given capacity
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Cache capacity must be > 0");

        Self {
            capacity,
            cache: BTreeMap::new(),
            access_counter: 0,
        }
    }

    /// Get value from cache (updates access time)
    ///
    /// Returns `Some(&V)` if key exists, `None` otherwise.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(entry) = self.cache.get_mut(key) {
            self.access_counter += 1;
            entry.access_time = self.access_counter;
            Some(&entry.value)
        } else {
            None
        }
    }

    /// Get mutable value from cache (updates access time)
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        if let Some(entry) = self.cache.get_mut(key) {
            self.access_counter += 1;
            entry.access_time = self.access_counter;
            Some(&mut entry.value)
        } else {
            None
        }
    }

    /// Check if key exists in cache (doesn't update access time)
    pub fn contains(&self, key: &K) -> bool {
        self.cache.contains_key(key)
    }

    /// Insert key-value pair into cache
    ///
    /// If cache is at capacity, evicts least recently used item.
    /// Returns the evicted entry if one was removed.
    pub fn insert(&mut self, key: K, value: V) -> Option<(K, V)> {
        let mut evicted = None;

        // Evict LRU if at capacity and key doesn't exist
        if self.cache.len() >= self.capacity && !self.cache.contains_key(&key) {
            evicted = self.evict_lru();
        }

        // Insert new entry
        self.access_counter += 1;
        let entry = CacheEntry {
            value,
            access_time: self.access_counter,
        };

        self.cache.insert(key, entry);

        evicted
    }

    /// Remove key from cache
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.cache.remove(key).map(|entry| entry.value)
    }

    /// Evict least recently used item
    ///
    /// Returns the evicted (key, value) pair.
    fn evict_lru(&mut self) -> Option<(K, V)> {
        // Find key with minimum access_time
        let lru_key = self
            .cache
            .iter()
            .min_by_key(|(_, entry)| entry.access_time)
            .map(|(k, _)| k.clone())?;

        let entry = self.cache.remove(&lru_key)?;
        Some((lru_key, entry.value))
    }

    /// Get current number of items in cache
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get cache capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all items from cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get hit rate (approximation based on access counter)
    pub fn access_count(&self) -> u64 {
        self.access_counter
    }

    /// Iterate over all keys (in no particular order)
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.cache.keys()
    }

    /// Iterate over all values (in no particular order)
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.cache.values().map(|entry| &entry.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_cache_creation() {
        let cache: LRUCache<i32, String> = LRUCache::new(10);
        assert_eq!(cache.capacity(), 10);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_lru_cache_insert_and_get() {
        let mut cache = LRUCache::new(3);

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&1), Some(&"one".to_string()));
        assert_eq!(cache.get(&2), Some(&"two".to_string()));
        assert_eq!(cache.get(&3), Some(&"three".to_string()));
        assert_eq!(cache.get(&4), None);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = LRUCache::new(2);

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        // Access key 1 to make it more recently used
        cache.get(&1);

        // Insert new item - should evict key 2 (LRU)
        let evicted = cache.insert(3, "three".to_string());

        assert!(evicted.is_some());
        let (evicted_key, evicted_val) = evicted.unwrap();
        assert_eq!(evicted_key, 2);
        assert_eq!(evicted_val, "two".to_string());

        // Cache should contain 1 and 3, not 2
        assert!(cache.contains(&1));
        assert!(!cache.contains(&2));
        assert!(cache.contains(&3));
    }

    #[test]
    fn test_lru_update_existing() {
        let mut cache = LRUCache::new(2);

        cache.insert(1, "one".to_string());
        cache.insert(1, "ONE".to_string()); // Update

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&1), Some(&"ONE".to_string()));
    }

    #[test]
    fn test_lru_remove() {
        let mut cache = LRUCache::new(3);

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let removed = cache.remove(&1);
        assert_eq!(removed, Some("one".to_string()));
        assert_eq!(cache.len(), 1);
        assert!(!cache.contains(&1));
    }

    #[test]
    fn test_lru_clear() {
        let mut cache = LRUCache::new(3);

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_lru_access_pattern() {
        let mut cache = LRUCache::new(3);

        // Fill cache
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        // Access pattern: 1, 2, 1, 3
        // Makes order: 3 (most recent), 1, 2 (least recent)
        cache.get(&1);
        cache.get(&2);
        cache.get(&1);
        cache.get(&3);

        // Insert new item - should evict 2
        let evicted = cache.insert(4, 400);
        assert_eq!(evicted, Some((2, 200)));

        // Verify final state
        assert!(cache.contains(&1));
        assert!(!cache.contains(&2));
        assert!(cache.contains(&3));
        assert!(cache.contains(&4));
    }

    #[test]
    fn test_lru_contains_no_update() {
        let mut cache = LRUCache::new(2);

        cache.insert(1, 100);
        cache.insert(2, 200);

        // contains() shouldn't update access time
        let initial_accesses = cache.access_count();
        cache.contains(&1);
        assert_eq!(cache.access_count(), initial_accesses);

        // get() should update access time
        cache.get(&1);
        assert!(cache.access_count() > initial_accesses);
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn test_zero_capacity_panics() {
        let _cache: LRUCache<i32, i32> = LRUCache::new(0);
    }
}
