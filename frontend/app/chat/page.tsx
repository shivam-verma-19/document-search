"use client";
import axios from "axios";
import { useState } from "react";

export default function Chat() {
  const [q, setQ] = useState("");
  const [a, setA] = useState("");

  const ask = async () => {
    const res = await axios.get(`/api/ask?q=${q}`);
    setA(res.data.answer);
  };

  return (
    <div>
      <input value={q} onChange={e=>setQ(e.target.value)} />
      <button onClick={ask}>Ask</button>
      <p>{a}</p>
    </div>
  );
}