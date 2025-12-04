// PMS9103M 串口解析：同时打印 质量浓度 + 粒子个数
// Windows (CH340/CP210x) 版
#include <stdio.h>
#include <windows.h>

#define FRAME_LEN 32

void log_to_csv(int pm1, int pm25, int pm10,
                int n03, int n05, int n10, int n25, int n50, int n100) {
    FILE *fp = fopen("pm_log2.csv", "a"); // 追加写入
    if (fp) {
        SYSTEMTIME st;
        GetLocalTime(&st);  // 获取本地时间（带毫秒）
        fprintf(fp, "%04d-%02d-%02d %02d:%02d:%02d.%03d,"
                    "%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                st.wYear, st.wMonth, st.wDay,
                st.wHour, st.wMinute, st.wSecond, st.wMilliseconds,
                pm1, pm25, pm10, n03, n05, n10, n25, n50, n100);
        fclose(fp);
    }
}


void print_timestamp_ms() {
    SYSTEMTIME st;
    GetLocalTime(&st);
    printf("[%04d-%02d-%02d %02d:%02d:%02d.%03d] ",
           st.wYear, st.wMonth, st.wDay,
           st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
}

static int read_exact(HANDLE h, unsigned char *buf, int n) {
    DWORD r = 0, got = 0;
    while (got < (DWORD)n) {
        if (!ReadFile(h, buf + got, n - got, &r, NULL)) return 0;
        if (r == 0) return 0;
        got += r;
    }
    return 1;
}

int main(void) {
    const char *port = "\\\\.\\COM6";            // 6右 
    HANDLE h;
    DCB dcb = {0};
    COMMTIMEOUTS to = {0};

    // 打开串口
    h = CreateFileA(port, GENERIC_READ|GENERIC_WRITE, 0, NULL,
                    OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE) { printf("无法打开端口 %s\n", port); return 1; }

    // 配置 9600 8N1
    dcb.DCBlength = sizeof(dcb);
    GetCommState(h, &dcb);
    dcb.BaudRate = CBR_9600;
    dcb.ByteSize = 8;
    dcb.Parity   = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    SetCommState(h, &dcb);

    // 超时
    to.ReadIntervalTimeout         = 50;
    to.ReadTotalTimeoutConstant    = 50;
    to.ReadTotalTimeoutMultiplier  = 10;
    SetCommTimeouts(h, &to);

    printf("开始读取...\n");
    unsigned char b, frame[FRAME_LEN];

    for (;;) {
        // 1) 找帧头 0x42 0x4D
        DWORD rd = 0;
        if (!ReadFile(h, &b, 1, &rd, NULL) || rd == 0) continue;
        if (b != 0x42) continue;
        if (!ReadFile(h, &b, 1, &rd, NULL) || rd == 0) continue;
        if (b != 0x4D) continue;

        frame[0] = 0x42; frame[1] = 0x4D;

        // 2) 读余下 30 字节
        if (!read_exact(h, frame + 2, FRAME_LEN - 2)) continue;

        // 3) 校验和（说明书：从起始符开始累加到 数据13低八位，取低16位）
        unsigned int sum = 0;
        for (int i = 0; i < 30; ++i) sum += frame[i];
        unsigned int crc = ((unsigned int)frame[30] << 8) | frame[31];
        int ok = ((sum & 0xFFFF) == crc);

        // 4) 解析（高字节在前）
        int pm1_env  = (frame[10] << 8) | frame[11];
        int pm25_env = (frame[12] << 8) | frame[13];
        int pm10_env = (frame[14] << 8) | frame[15];

        int n_03 = (frame[16] << 8) | frame[17]; // >0.3μm /0.1L
        int n_05 = (frame[18] << 8) | frame[19]; // >0.5μm
        int n_10 = (frame[20] << 8) | frame[21]; // >1.0μm
        int n_25 = (frame[22] << 8) | frame[23]; // >2.5μm
        int n_50 = (frame[24] << 8) | frame[25]; // >5.0μm
        int n_100= (frame[26] << 8) | frame[27]; // >10μm

        // 6) 打印“质量浓度 + 粒子个数”
        print_timestamp_ms();
        printf("\n");
        printf("大气值: PM1.0=%d μg/m3  PM2.5=%d μg/m3  PM10=%d μg/m3\n",
               pm1_env, pm25_env, pm10_env);
        printf("粒子个数(每0.1L): >0.3=%d  >0.5=%d  >1.0=%d  >2.5=%d  >5=%d  >10=%d\n\n",
               n_03, n_05, n_10, n_25, n_50, n_100);
    
		// 写入 CSV 文件
		log_to_csv(pm1_env, pm25_env, pm10_env, n_03, n_05, n_10, n_25, n_50, n_100);
	
	
	}

    CloseHandle(h);
    return 0;
}
