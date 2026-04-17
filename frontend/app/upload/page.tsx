"use client";
import axios from "axios";
import { useState } from "react";

export default function Upload() {
  const [file, setFile] = useState<any>();

  const upload = async () => {
    const fd = new FormData();
    fd.append("file", file);
    await axios.post("/api/upload", fd);
  };

  return (
    <div>
      <input type="file" onChange={e=>setFile(e.target.files?.[0])}/>
      <button onClick={upload}>Upload</button>
    </div>
  );
}