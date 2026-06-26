"use client";

import { MathJax, MathJaxContext, type MathJax3Config } from "better-react-mathjax";
import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { edges, nodes, territories, type MapEdge, type MapNode, type Marker } from "./map-data";

const NODE_WIDTH = 245;
const LABEL_HEIGHT = 54;
const MAP_WIDTH = 2310;
const MAP_HEIGHT = 1120;
const MAP_LEFT = 92;
const MAP_TOP = 150;
const COL_STEP = 270;
const ROW_STEP = 122;

const mathJaxConfig: MathJax3Config = {
  tex: {
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    displayMath: [
      ["$$", "$$"],
      ["\\[", "\\]"],
    ],
    processEscapes: true,
    processEnvironments: true,
    macros: {
      R: "\\mathbb{R}",
      C: "\\mathbb{C}",
      E: "\\mathbb{E}",
      Sph: "\\mathbb{S}",
      Ball: "\\mathbb{B}",
      Mob: "\\operatorname{Mob}",
      SO: "\\operatorname{SO}",
      grad: "\\operatorname{grad}",
      hyp: "\\mathrm{hyp}",
      euc: "\\mathrm{euc}",
      dd: "\\,\\mathrm{d}",
      vct: ["\\mathbf{#1}", 1],
      norm: ["\\left\\lVert #1 \\right\\rVert", 1],
      inner: ["\\left\\langle #1,#2 \\right\\rangle", 2],
    },
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"],
  },
};

type PositionedNode = MapNode & {
  x: number;
  y: number;
};

type CustomStyle = CSSProperties & {
  "--territory-color"?: string;
};

const markerGlyph: Record<Marker, string> = {
  visualize: "*",
  play: "o",
};

function buildPositionedNodes() {
  return nodes.map((node) => ({
    ...node,
    x: MAP_LEFT + node.col * COL_STEP,
    y: MAP_TOP + node.row * ROW_STEP,
  }));
}

function territoryFor(id: string) {
  return territories.find((territory) => territory.id === id) ?? territories[0];
}

function splitTitle(title: string) {
  if (title.length <= 24) return [title];
  const words = title.split(" ");
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length > 24 && current) {
      lines.push(current);
      current = word;
    } else {
      current = next;
    }
  }
  if (current) lines.push(current);
  return lines.slice(0, 2);
}

function nodeAnchor(node: PositionedNode, vertical: "top" | "bottom" | "center") {
  const x = node.x + NODE_WIDTH / 2;
  if (vertical === "top") return { x, y: node.y + 2 };
  if (vertical === "bottom") return { x, y: node.y + LABEL_HEIGHT + 2 };
  return { x, y: node.y + LABEL_HEIGHT / 2 };
}

function sameLevelPath(source: PositionedNode, target: PositionedNode, edge: MapEdge) {
  const start = nodeAnchor(source, "top");
  const end = nodeAnchor(target, "top");
  const left = start.x <= end.x ? start : end;
  const right = start.x <= end.x ? end : start;
  const dx = Math.max(40, Math.abs(right.x - left.x));
  const rx = dx / 2;
  const ry = Math.max(60, dx * 0.34 + Math.abs(edge.lane ?? 0) * 28);
  const sweep = edge.arcSide === "down" ? 1 : 0;
  return `M ${left.x} ${left.y} A ${rx} ${ry} 0 0 ${sweep} ${right.x} ${right.y}`;
}

function betweenLevelPath(source: PositionedNode, target: PositionedNode, edge: MapEdge) {
  const downward = target.y >= source.y;
  const start = nodeAnchor(source, downward ? "bottom" : "top");
  const end = nodeAnchor(target, downward ? "top" : "bottom");
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lane = edge.lane ?? 0;
  const laneOffset = lane * 92;
  const c1 = {
    x: start.x + dx * 0.18 + laneOffset,
    y: start.y + dy * 0.12,
  };
  const c2 = {
    x: start.x + dx * 0.82 - laneOffset,
    y: start.y + dy * 0.9,
  };
  return `M ${start.x} ${start.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${end.x} ${end.y}`;
}

function edgePath(edge: MapEdge, nodeMap: Map<string, PositionedNode>) {
  const source = nodeMap.get(edge.source);
  const target = nodeMap.get(edge.target);
  if (!source || !target) return "";
  if (edge.levelRelation === "same" || source.row === target.row) {
    return sameLevelPath(source, target, edge);
  }
  return betweenLevelPath(source, target, edge);
}

function relatedEdges(nodeId: string) {
  return edges.filter((edge) => edge.source === nodeId || edge.target === nodeId);
}

function initialNodeFromUrl() {
  if (typeof window === "undefined") return null;
  const nodeId = new URLSearchParams(window.location.search).get("node");
  return nodeId && nodes.some((node) => node.id === nodeId) ? nodeId : null;
}

function initialPanelOpen() {
  if (typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).get("open") === "1";
}

export default function GraphClient() {
  const positionedNodes = useMemo(() => buildPositionedNodes(), []);
  const nodeMap = useMemo(
    () => new Map(positionedNodes.map((node) => [node.id, node])),
    [positionedNodes],
  );
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(() => initialNodeFromUrl());
  const [panelOpen, setPanelOpen] = useState(() => initialPanelOpen());

  const activeId = hoveredId ?? selectedId;
  const activeNode = activeId ? nodeMap.get(activeId) ?? null : null;
  const panelNode = selectedId ? nodeMap.get(selectedId) ?? null : null;

  const activeEdges = useMemo(
    () => (activeId ? relatedEdges(activeId) : []),
    [activeId],
  );

  const connectedIds = useMemo(() => {
    const ids = new Set<string>();
    if (!activeId) return ids;
    ids.add(activeId);
    for (const edge of activeEdges) {
      ids.add(edge.source);
      ids.add(edge.target);
    }
    return ids;
  }, [activeEdges, activeId]);

  const incoming = panelNode ? edges.filter((edge) => edge.target === panelNode.id) : [];
  const outgoing = panelNode ? edges.filter((edge) => edge.source === panelNode.id) : [];

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      setPanelOpen(false);
      setSelectedId(null);
      window.history.replaceState(null, "", window.location.pathname);
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  function openNode(nodeId: string) {
    setSelectedId(nodeId);
    setPanelOpen(true);
    window.history.replaceState(null, "", `?node=${nodeId}&open=1`);
  }

  function closePanel() {
    setPanelOpen(false);
    setSelectedId(null);
    window.history.replaceState(null, "", window.location.pathname);
  }

  return (
    <MathJaxContext version={3} config={mathJaxConfig} src="/vendor/mathjax/tex-mml-chtml.js">
      <main className="map-page">
      <header className="map-header">
        <div>
          <h1>LMS Exact-Gauge Map</h1>
          <div className="map-legend" aria-label="Topic marker legend">
            <span>
              <b>*</b> Visualize It
            </span>
            <span>
              <b>o</b> Play With It
            </span>
          </div>
        </div>
      </header>

      <section className="map-scroll" aria-label="Interactive LMS framework map">
        <svg
          className={activeId ? "math-map has-active" : "math-map"}
          width={MAP_WIDTH}
          height={MAP_HEIGHT}
          viewBox={`0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`}
          role="img"
          aria-label="Map of LMS foundations, exact inversion, state construction, moving frames, continuum layer, and passages"
        >
          <g className="active-arcs">
            {activeEdges.map((edge) => {
              const path = edgePath(edge, nodeMap);
              if (!path) return null;
              return (
                <path
                  key={`${edge.source}-${edge.target}`}
                  className={`active-arc ${edge.levelRelation}`}
                  d={path}
                />
              );
            })}
          </g>

          <g className="node-layer">
            {positionedNodes.map((node) => {
              const territory = territoryFor(node.territory);
              const isActive = activeId === node.id;
              const isConnected = connectedIds.has(node.id);
              const isDimmed = Boolean(activeId) && !isConnected;
              const lines = splitTitle(node.title);
              const style: CustomStyle = { "--territory-color": territory.color };

              return (
                <g
                  key={node.id}
                  className={[
                    "map-node",
                    node.level === 0 ? "territory-node" : "",
                    isActive ? "is-active" : "",
                    isConnected ? "is-connected" : "",
                    isDimmed ? "is-dimmed" : "",
                  ].join(" ")}
                  style={style}
                  transform={`translate(${node.x} ${node.y})`}
                  tabIndex={0}
                  role="button"
                  aria-pressed={isActive}
                  aria-label={`Open ${node.title}`}
                  onMouseEnter={() => setHoveredId(node.id)}
                  onMouseLeave={() => setHoveredId(null)}
                  onFocus={() => setHoveredId(node.id)}
                  onBlur={() => setHoveredId(null)}
                  onClick={() => openNode(node.id)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      openNode(node.id);
                    }
                  }}
                >
                  <rect className="hit-area" x="0" y="-16" width={NODE_WIDTH} height="76" rx="0" />
                  <line className="node-rule" x1="0" y1="0" x2={NODE_WIDTH} y2="0" />
                  <text className="node-title" x="0" y="24">
                    {lines.map((line, index) => (
                      <tspan key={line} x="0" dy={index === 0 ? 0 : 18}>
                        {line}
                      </tspan>
                    ))}
                  </text>
                  {node.markers?.map((marker, index) => (
                    <text
                      key={`${node.id}-${marker}`}
                      className={`node-marker ${marker}`}
                      x={NODE_WIDTH - 10 - index * 18}
                      y="24"
                    >
                      {markerGlyph[marker]}
                    </text>
                  ))}
                </g>
              );
            })}
          </g>
        </svg>
      </section>

      {panelOpen && panelNode && (
        <div className="topic-overlay" role="dialog" aria-modal="true" aria-labelledby="topic-title">
          <button className="overlay-backdrop" type="button" aria-label="Close topic panel" onClick={closePanel} />
          <aside
            className="topic-panel"
            style={{ "--territory-color": territoryFor(panelNode.territory).color } as CustomStyle}
          >
            <div className="topic-panel-header">
              <div>
                <p className="topic-kicker">{territoryFor(panelNode.territory).title}</p>
                <h2 id="topic-title">
                  <MathJax inline dynamic>
                    {panelNode.title}
                  </MathJax>
                </h2>
              </div>
              <button className="close-button" type="button" onClick={closePanel} aria-label="Close topic panel">
                x
              </button>
            </div>

            <p className="topic-teaser">
              <MathJax inline dynamic>
                {panelNode.teaser}
              </MathJax>
            </p>

            <div className="topic-status-row">
              <span>{panelNode.status}</span>
              {panelNode.markers?.map((marker) => (
                <span key={marker}>
                  {markerGlyph[marker]} {marker === "visualize" ? "Visualize It" : "Play With It"}
                </span>
              ))}
            </div>

            <section className="topic-section">
              {panelNode.detail.map((paragraph) => (
                <p key={paragraph}>
                  <MathJax inline dynamic>
                    {paragraph}
                  </MathJax>
                </p>
              ))}
            </section>

            {panelNode.formulas.length > 0 && (
              <section className="topic-section">
                <h3>Equations</h3>
                {panelNode.formulas.map((formula) => (
                  <div className="topic-equation" key={formula}>
                    <MathJax dynamic>{formula}</MathJax>
                  </div>
                ))}
              </section>
            )}

            {(incoming.length > 0 || outgoing.length > 0) && (
              <section className="topic-section relation-grid">
                <div>
                  <h3>Depends on</h3>
                  {incoming.length ? (
                    incoming.map((edge) => (
                      <button key={`${edge.source}-${edge.target}`} type="button" onClick={() => openNode(edge.source)}>
                        <span className="relation-title">
                          <MathJax inline dynamic>
                            {nodeMap.get(edge.source)?.title}
                          </MathJax>
                        </span>
                        <small>
                          <MathJax inline dynamic>
                            {edge.label}
                          </MathJax>
                        </small>
                      </button>
                    ))
                  ) : (
                    <p>
                      <MathJax inline>No incoming edge in this map.</MathJax>
                    </p>
                  )}
                </div>
                <div>
                  <h3>Leads to</h3>
                  {outgoing.length ? (
                    outgoing.map((edge) => (
                      <button key={`${edge.source}-${edge.target}`} type="button" onClick={() => openNode(edge.target)}>
                        <span className="relation-title">
                          <MathJax inline dynamic>
                            {nodeMap.get(edge.target)?.title}
                          </MathJax>
                        </span>
                        <small>
                          <MathJax inline dynamic>
                            {edge.label}
                          </MathJax>
                        </small>
                      </button>
                    ))
                  ) : (
                    <p>
                      <MathJax inline>No outgoing edge in this map.</MathJax>
                    </p>
                  )}
                </div>
              </section>
            )}

            <section className="topic-section source-list">
              <h3>Sources</h3>
              {panelNode.sources.map((source) => (
                <p key={source}>{source}</p>
              ))}
            </section>
          </aside>
        </div>
      )}

      <div className="active-readout" aria-live="polite">
        {activeNode ? activeNode.title : " "}
      </div>
      </main>
    </MathJaxContext>
  );
}
