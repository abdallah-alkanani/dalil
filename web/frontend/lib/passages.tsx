"use client";

import { createContext, useContext } from "react";
import type { Passage } from "./types";

/**
 * The passages retrieved for one answer. Generated components refer to passages
 * only by number and read their text from here, so nothing the model writes can
 * stand in for a source's wording.
 */
export interface PassagesContextValue {
  passages: Passage[];
  open: (n: number, highlight?: string) => void;
}

export const PassagesContext = createContext<PassagesContextValue>({ passages: [], open: () => {} });

export function usePassages(): PassagesContextValue {
  return useContext(PassagesContext);
}

export function usePassage(n: number | undefined): Passage | undefined {
  const { passages } = usePassages();
  return n == null ? undefined : passages.find((p) => p.n === n);
}
