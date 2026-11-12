import { NextRequest, NextResponse } from "next/server";
import { proxyToBackend } from "../../../lib/backend-proxy";

export async function GET(request: NextRequest) {
  return proxyToBackend(request, []);
}

export async function HEAD(request: NextRequest) {
  return proxyToBackend(request, []);
}

export async function POST(request: NextRequest) {
  return proxyToBackend(request, []);
}

export async function OPTIONS() {
  return new NextResponse(null, { status: 204 });
}
