
import java.util.*;

public class MST {
	
	private static class Edge {
		int u;
		int v;
		float dist;
	}
	
	private int n;
	private boolean edges[][];
	private double sum = 0;
	
	public MST(float[][] distance) {
		n = distance.length;
		edges = new boolean[n][n];
		boolean[] visited = new boolean[n];
		visited[0] = true;
		for (int i = 1; i < n; ++i) {
			double best = 999999999;
			int u = 0;
			int v = 0;
			for (int j = 0; j < n; ++j) for (int k = 0; k < n; ++k) {
				if (visited[j] && !visited[k]) {
					if (distance[j][k] < best) {
						best = distance[j][k];
						u = j;
						v = k;
					}
				}
			}
			edges[u][v] = true;
			edges[v][u] = true;
			visited[v] = true;
			sum += best;
		}
	}
	
	public double getSum() {
		return sum;
	}
	
	private boolean[] sequenceVisited;
	private int[] seq;
	private int offset;
	
	public void fillSequence(int[] sequence) {
		sequenceVisited = new boolean[n];
		seq = sequence;
		offset = 0;
		dfs(0);
	}
	
	private void dfs(int u) {
		seq[offset++] = u;
		sequenceVisited[u] = true;
		for (int v = 0; v < n; ++v) {
			if (edges[u][v] && !sequenceVisited[v]) {
				dfs(v);
			}
		}
	}
	
}

