import { useSyncExternalStore } from "react"

// A value that components subscribe to one by one, so a fast-changing value (live stats at 10 Hz)
// re-renders only what shows it, not the whole page.
export interface Store<T> {
  get: () => T
  set: (v: T) => void
  subscribe: (fn: () => void) => () => void
}

export function createStore<T>(initial: T): Store<T> {
  let value = initial
  const subs = new Set<() => void>()
  return {
    get: () => value,
    set: (v) => {
      if (Object.is(v, value)) return
      value = v
      subs.forEach((fn) => fn())
    },
    subscribe: (fn) => {
      subs.add(fn)
      return () => subs.delete(fn)
    },
  }
}

export function useStore<T>(store: Store<T>): T
export function useStore<T, S>(store: Store<T>, select: (v: T) => S): S
export function useStore<T, S>(store: Store<T>, select?: (v: T) => S) {
  return useSyncExternalStore<T | S>(store.subscribe, select ? () => select(store.get()) : store.get)
}
