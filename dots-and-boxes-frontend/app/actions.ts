"use server";

const API_BASE_URL = process.env.API_BASE_URL || "http://127.0.0.1:5000";

export async function initializeGame(size: number) {
  try {
    console.log("Initializing game with size:", size);
    console.log("API_BASE_URL:", API_BASE_URL);
    const response = await fetch(`${API_BASE_URL}/start`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ size }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `HTTP error! status: ${response.status}, message: ${errorText}`
      );
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error initializing game:", error);
    return {
      gameState: null,
      error:
        error instanceof Error
          ? error.message
          : "An unknown error occurred while initializing the game",
    };
  }
}

export async function makeMove(row: number, col: number, sessionId: string) {
  try {
    const response = await fetch(`${API_BASE_URL}/move`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ row, col, session_id: sessionId }),
    });

    
    if (!response.ok) {
      console.error("Failed to make move:", response.statusText);
      console.error("Response:", await response.json());
      throw new Error("Failed to make move");
    }

    const data = await response.json();
    console.log("move made:", data);
    return data;
  } catch (error) {
    console.error("Error making move:", error);
    return { gameState: null, error: "Failed to make move" };
  }
}
