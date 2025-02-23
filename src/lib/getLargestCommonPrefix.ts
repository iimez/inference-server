

export function getLargestCommonPrefix(str1: string, str2: string): string {
	const minLength = Math.min(str1.length, str2.length);
	let prefixLength = 0;
	
	while (prefixLength < minLength && str1[prefixLength] === str2[prefixLength]) {
			prefixLength++;
	}
	
	return str1.slice(0, prefixLength);
}