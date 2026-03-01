/*
Sudoku - a fast Java Sudoku game creation library.
Copyright (C) 2017-2018  Stephan Fuhrmann

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Library General Public
License as published by the Free Software Foundation; either
version 2 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Library General Public License for more details.

You should have received a copy of the GNU Library General Public
License along with this library; if not, write to the
Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
Boston, MA  02110-1301, USA.
 */
package de.sfuhrm.sudoku;

import sun.util.calendar.CalendarUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Solves a partially filled Sudoku. Can find multiple solutions if they are
 * there.
 *
 * @author Stephan Fuhrmann
 */
public final class Solver {

    /**
     * Current working copy.
     */
    private final CachedGameMatrixImpl riddle;

    /**
     * The possible solutions for this riddle.
     */
    private final List<GameMatrix> possibleSolutions;

    /** The default limit.
     * @see #limit
     */
    public static final int DEFAULT_LIMIT = 1;

    /**
     * The maximum number of solutions to search.
     */
    private int limit;

    /**
     * Counter for recursive calls on bactrack algorithm
     */
    private static long recursive_calls=0;
    public static long start;

    private static final long DEFAULT_STATS_STEP = 1000000;

    /** Número màxim de fils que permetrem */
    private static final int MAX_THREADS =
            Runtime.getRuntime().availableProcessors() * 2;

    /** Comptador de fils actius */
    private final AtomicInteger activeThreads = new AtomicInteger(0);

    /** Per parar tots els fils quan trobem una solució */
    private volatile boolean solutionFound = false;
    /**
     * Creates a solver for the given riddle.
     *
     * @param solveMe the riddle to solve.
     */
    public Solver(final GameMatrix solveMe) {
        Objects.requireNonNull(solveMe, "solveMe is null");
        limit = DEFAULT_LIMIT;
        riddle = new CachedGameMatrixImpl(solveMe.getSchema());
        riddle.setAll(solveMe.getArray());
        possibleSolutions = new ArrayList<>();
    }

    public static synchronized long getRecursive_calls() {
        return recursive_calls;
    }

    public static synchronized void setRecursive_calls(long recursive_calls) {
        Solver.recursive_calls = recursive_calls;
    }

    public static synchronized long incRecursive_calls() {
        return ++Solver.recursive_calls;
    }

    /** Set the limit for maximum results.
     * @param set the new limit.
     */
    public void setLimit(final int set) {
        this.limit = set;
    }

    /**
     * Solves the Sudoku problem.
     *
     * @return the found solutions. Should be only one.
     */
    public List<GameMatrix> solve() {
        start = System.currentTimeMillis();
        possibleSolutions.clear();
        int freeCells = riddle.getSchema().getTotalFields()
                - riddle.getSetCount();

        backtrack(freeCells, new CellIndex(), riddle);

        long end = System.currentTimeMillis();
        System.out.printf("[%3.3f] SUDOKU DONE. Recursive Calls: %d. Free Cells: %d. Solutions Found: %d.\n\n",(end-start)/1000.0, this.getRecursive_calls(), freeCells, possibleSolutions.size());

        return Collections.unmodifiableList(possibleSolutions);
    }

    private class BacktrackTaks implements Runnable {
        private final int freeCells;
        private final CellIndex minimumCell;
        private final CachedGameMatrixImpl board;

        private BacktrackTaks(int freeCells, CellIndex minimumCell, CachedGameMatrixImpl board) {
            this.freeCells = freeCells;
            this.minimumCell = minimumCell;
            this.board = board;
        }

        @Override
        public void run() {
            try {
                backtrack(freeCells, minimumCell, board);
            } finally {
                activeThreads.decrementAndGet();
            }
        }

    }
    /**
     * Solves a Sudoku using backtracking.
     *
     * @param freeCells number of free cells, abort criterion.
     * @param minimumCell coordinates to the so-far found minimum cell.
     * @return the total number of solutions.
     */

    private int backtrack(final int freeCells, final CellIndex minimumCell, CachedGameMatrixImpl board) {
        assert freeCells >= 0 : "freeCells is negative";

        if ((this.incRecursive_calls()%DEFAULT_STATS_STEP)==0) {
            long end = System.currentTimeMillis();
            System.out.printf("[%3.3f] Recursive Calls: %d. Free Cells: %d. Solutions Found: %d.\n",(end-start)/1000.0, this.getRecursive_calls(), freeCells, possibleSolutions.size());
        }
        // don't recurse further if already at limit
        if (possibleSolutions.size() >= limit) {
            return 0;
        }

        // just one result, we have no more to choose
        if (freeCells == 0) {
            GameMatrix gmi = new GameMatrixImpl(board.getSchema());
            gmi.setAll(board.getArray());
            possibleSolutions.add(gmi);

            return 1;
        }

        GameMatrixImpl.FreeCellResult freeCellResult =
                board.findLeastFreeCell(minimumCell);
        if (freeCellResult != GameMatrixImpl.FreeCellResult.FOUND) {
            // no solution
            return 0;
        }

        int result = 0;
        int minimumRow = minimumCell.row;
        int minimumColumn = minimumCell.column;
        int minimumFree = board.getFreeMask(minimumRow, minimumColumn);
        int minimumBits = Integer.bitCount(minimumFree);

        // else we are done
        // now try each number
        for (int bit = 0; bit < minimumBits; bit++) {
            int index = Creator.getSetBitOffset(minimumFree, bit);
            assert index > 0;

            if (freeCells > 40 && activeThreads.get() < MAX_THREADS && !solutionFound) {

                activeThreads.incrementAndGet();

                // Fem una copia del fitxer
                CachedGameMatrixImpl childCopy = board.clone();
                childCopy.set(minimumRow, minimumColumn, (byte) index);
                // inicialitzem el fil
                Thread t = new Thread(
                        new BacktrackTaks(freeCells - 1, new CellIndex(), childCopy)
                );
                t.start();

            } else {

                // Asignamos número index a la celda
                board.set(minimumRow, minimumColumn, (byte) index);
                int resultCount = backtrack(freeCells - 1, minimumCell, board);
                result += resultCount;
            }
        }
        // Antes de volver marcamos la celda como no asignada
        board.set(minimumRow,
                minimumColumn,
                board.getSchema().getUnsetValue());

        return result;
    }


}
